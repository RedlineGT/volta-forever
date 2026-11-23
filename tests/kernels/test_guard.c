/*
 * test_guard.c: a kernel that the layer refused, of which the program holds a handle that the layer
 * did not give, is not launched. The handle is taken here from NVIDIA's driver itself, which the
 * proxy has loaded, as a library takes it through a private interface of the driver (cuFFT takes
 * the function of each module it loads from an entry of a table of cuGetExportTable). Every launch
 * function, a kernel node of a graph and the questions about the kernel must return 218, and the
 * GPU must not stop: the context still runs a kernel afterwards.
 *
 *   test_guard module    load with cuModuleLoadData; the function from NVIDIA's cuModuleGetFunction
 *   test_guard library   load with cuLibraryLoadData; when the library is loaded whole (VF_LAZY=0),
 *                        the kernel from NVIDIA's cuLibraryGetKernel and its function from NVIDIA's
 *                        cuKernelGetFunction; in both cases, the function of the library's module
 *                        from NVIDIA's cuModuleGetFunction
 *   test_guard contexts  (VF_LAZY=0) the function of the library's refused kernel in context A, from
 *                        NVIDIA's cuKernelGetFunction, before the program asked the layer for anything
 *                        of the library in A, asked about while no context is current, then while
 *                        another context is; then asked about in A, where it must be refused whatever
 *                        was answered elsewhere. Nothing is launched but add_one, at the end.
 *
 * Build: gcc -O2 -Wall -o test_guard test_guard.c -ldl
 * Run:   DRVRUN_LIBCUDA=<proxy> CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> ./test_guard module
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef unsigned long long CUdeviceptr;

/* refused_by_layer uses barrier.cluster.arrive, of sm_90, for which the proxy has no sequence: it
   makes the function trap and refuses the kernel. */
static const char module_text[] =
    ".version 9.0\n"
    ".target sm_75\n"
    ".address_size 64\n"
    ".visible .entry add_one(.param .u64 p)\n"
    "{\n"
    "    .reg .u64 %rd<2>;\n"
    "    .reg .u32 %r<2>;\n"
    "    ld.param.u64 %rd0, [p];\n"
    "    cvta.to.global.u64 %rd1, %rd0;\n"
    "    ld.global.u32 %r0, [%rd1];\n"
    "    add.u32 %r1, %r0, 1;\n"
    "    st.global.u32 [%rd1], %r1;\n"
    "    ret;\n"
    "}\n"
    ".visible .entry refused_by_layer(.param .u64 p)\n"
    "{\n"
    "    barrier.cluster.arrive;\n"
    "    barrier.cluster.wait;\n"
    "    ret;\n"
    "}\n";

/* CUlaunchConfig, CUDA_KERNEL_NODE_PARAMS_v2 (v1: its first fields), CUgraphNodeParams. */
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
struct graph_node {
    int type, reserved[3];
    struct kernel_node kernel;
    char pad[232 - sizeof(struct kernel_node)];
    long long reserved2;
};

static void *cuda, *nvidia;
#define LAYER(name) dlsym(cuda, name)
#define NVIDIA(name) dlsym(nvidia, name)
static int failed, library_mode;
static void *loaded;
static CUdeviceptr word;
static void *args[] = {&word};

static void check(const char *what, int ok) {
    printf("  %s  %s\n", ok ? "ok    " : "WRONG ", what);
    failed |= !ok;
}

/* NVIDIA's library, as the proxy loaded it: the loaded object named libcuda.so.<something> that is
   not the proxy. */
static const char *proxy_path;
static char nvidia_path[PATH_MAX];
static int find_nvidia(struct dl_phdr_info *info, size_t size, void *data) {
    (void)size; (void)data;
    char real[PATH_MAX];
    const char *base = strrchr(info->dlpi_name, '/');
    if (!base || strncmp(base + 1, "libcuda.so.", 11)) return 0;
    if (realpath(info->dlpi_name, real) && strcmp(real, proxy_path) == 0) return 0;
    snprintf(nvidia_path, sizeof nvidia_path, "%s", info->dlpi_name);
    return 1;
}

static unsigned read_word(void) {
    CUresult (*from_device)(void *, CUdeviceptr, size_t) = LAYER("cuMemcpyDtoH_v2");
    unsigned value = 0xdead0000u;
    from_device(&value, word, sizeof value);
    return value;
}

/* Launch the layer's add_one on the word holding `in`; what it leaves there, or 0xdead0000 + error. */
static unsigned add_one(unsigned in) {
    CUresult (*to_device)(CUdeviceptr, const void *, size_t) = LAYER("cuMemcpyHtoD_v2");
    CUresult (*launch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **,
                       void **) = LAYER("cuLaunchKernel");
    CUresult (*synchronize)(void) = LAYER("cuCtxSynchronize");
    void *function = NULL, *kernel = NULL;
    CUresult r;
    if (library_mode) {
        CUresult (*get)(void **, void *, const char *) = LAYER("cuLibraryGetKernel");
        CUresult (*as_function)(void **, void *) = LAYER("cuKernelGetFunction");
        r = get(&kernel, loaded, "add_one");
        if (!r) r = as_function(&function, kernel);
    } else {
        CUresult (*get)(void **, void *, const char *) = LAYER("cuModuleGetFunction");
        r = get(&function, loaded, "add_one");
    }
    if (!r) r = to_device(word, &in, sizeof in);
    if (!r) r = launch(function, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL);
    if (!r) r = synchronize();
    return r ? 0xdead0000u + r : read_word();
}

/* Every call of the layer that would launch the function `f`, record it in a graph or ask about it:
   each must return 218. `good` is a function that runs, for the graph that is launched. */
static void refused_function(const char *what, void *f, void *good) {
    char line[200];
#define EXPECT(call, expr)                                                                \
    do {                                                                                  \
        CUresult got_ = (expr);                                                           \
        snprintf(line, sizeof line, "%s: %s refused with 218 (got %d)", what, call, got_); \
        check(line, got_ == 218);                                                         \
    } while (0)
    int value = 0;
    size_t offset = 0, size = 0;
    CUresult (*attribute)(int *, int, void *) = LAYER("cuFuncGetAttribute");
    EXPECT("cuFuncGetAttribute", attribute(&value, 0, f));
    CUresult (*param_info)(void *, size_t, size_t *, size_t *) = LAYER("cuFuncGetParamInfo");
    EXPECT("cuFuncGetParamInfo", param_info(f, 0, &offset, &size));
    CUresult (*occupancy)(int *, void *, int, size_t) = LAYER("cuOccupancyMaxActiveBlocksPerMultiprocessor");
    EXPECT("cuOccupancyMaxActiveBlocksPerMultiprocessor", occupancy(&value, f, 32, 0));
    typedef CUresult (*launch_t)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *,
                                 void **, void **);
    launch_t launch = LAYER("cuLaunchKernel"), launch_ptsz = LAYER("cuLaunchKernel_ptsz");
    EXPECT("cuLaunchKernel", launch(f, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL));
    EXPECT("cuLaunchKernel_ptsz", launch_ptsz(f, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL));
    /* As the CUDA runtime takes it. */
    CUresult (*proc)(const char *, void **, int, unsigned long long, int *) = LAYER("cuGetProcAddress_v2");
    void *looked_up = NULL;
    int status = 0;
    if (proc("cuLaunchKernel", &looked_up, 13000, 0, &status) == 0 && looked_up)
        EXPECT("cuLaunchKernel from cuGetProcAddress", ((launch_t)looked_up)(f, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL));
    else
        check("cuGetProcAddress gives cuLaunchKernel", 0);
    struct launch_config config = {{1, 1, 1}, {1, 1, 1}, 0, NULL, NULL, 0};
    CUresult (*launch_ex)(const void *, void *, void **, void **) = LAYER("cuLaunchKernelEx");
    CUresult (*launch_ex_ptsz)(const void *, void *, void **, void **) = LAYER("cuLaunchKernelEx_ptsz");
    EXPECT("cuLaunchKernelEx", launch_ex(&config, f, args, NULL));
    EXPECT("cuLaunchKernelEx_ptsz", launch_ex_ptsz(&config, f, args, NULL));
    CUresult (*cooperative)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *,
                            void **) = LAYER("cuLaunchCooperativeKernel");
    EXPECT("cuLaunchCooperativeKernel", cooperative(f, 1, 1, 1, 1, 1, 1, 0, NULL, args));
    CUresult (*old_launch)(void *) = LAYER("cuLaunch");
    CUresult (*grid)(void *, int, int) = LAYER("cuLaunchGrid");
    CUresult (*grid_async)(void *, int, int, void *) = LAYER("cuLaunchGridAsync");
    EXPECT("cuLaunch", old_launch(f));
    EXPECT("cuLaunchGrid", grid(f, 1, 1));
    EXPECT("cuLaunchGridAsync", grid_async(f, 1, 1, NULL));

    /* Graphs: a node with the kernel is not made, a node is not given it, and the graph with a node
       that runs is launched. */
    CUresult (*create)(void **, unsigned) = LAYER("cuGraphCreate");
    CUresult (*add_v1)(void **, void *, void **, size_t, const void *) = LAYER("cuGraphAddKernelNode");
    CUresult (*add)(void **, void *, void **, size_t, const void *) = LAYER("cuGraphAddKernelNode_v2");
    CUresult (*add_node)(void **, void *, void **, size_t, void *) = LAYER("cuGraphAddNode");
    CUresult (*set)(void *, const void *) = LAYER("cuGraphKernelNodeSetParams_v2");
    CUresult (*instantiate)(void **, void *, unsigned long long) = LAYER("cuGraphInstantiateWithFlags");
    CUresult (*exec_set)(void *, void *, const void *) = LAYER("cuGraphExecKernelNodeSetParams_v2");
    CUresult (*graph_launch)(void *, void *) = LAYER("cuGraphLaunch");
    CUresult (*synchronize)(void) = LAYER("cuCtxSynchronize");
    CUresult (*exec_destroy)(void *) = LAYER("cuGraphExecDestroy");
    CUresult (*destroy)(void *) = LAYER("cuGraphDestroy");
    CUresult (*to_device)(CUdeviceptr, const void *, size_t) = LAYER("cuMemcpyHtoD_v2");
    void *graph = NULL, *node = NULL, *exec = NULL;
    struct kernel_node bad = {f, {1, 1, 1}, {1, 1, 1}, 0, args, NULL, NULL, NULL};
    struct kernel_node runs = {good, {1, 1, 1}, {1, 1, 1}, 0, args, NULL, NULL, NULL};
    struct graph_node any = {0};
    any.kernel = bad;
    check("cuGraphCreate", create(&graph, 0) == 0);
    EXPECT("cuGraphAddKernelNode", add_v1(&node, graph, NULL, 0, &bad));
    EXPECT("cuGraphAddKernelNode_v2", add(&node, graph, NULL, 0, &bad));
    EXPECT("cuGraphAddNode", add_node(&node, graph, NULL, 0, &any));
    check("a node that runs is added", add(&node, graph, NULL, 0, &runs) == 0);
    EXPECT("cuGraphKernelNodeSetParams_v2", set(node, &bad));
    check("the graph is instantiated", instantiate(&exec, graph, 0) == 0);
    EXPECT("cuGraphExecKernelNodeSetParams_v2", exec_set(exec, node, &bad));
    unsigned seven = 7;
    to_device(word, &seven, sizeof seven);
    CUresult r = graph_launch(exec, NULL);
    if (!r) r = synchronize();
    snprintf(line, sizeof line, "%s: the graph runs its node that runs (%d, word %u)", what, r, read_word());
    check(line, r == 0 && read_word() == 8);
    exec_destroy(exec);
    destroy(graph);
#undef EXPECT
}

/* The calls that take a library's kernel `k`. */
static void refused_kernel(const char *what, void *k) {
    char line[200];
    int value = 0;
    CUresult (*attribute)(int *, int, void *, int) = LAYER("cuKernelGetAttribute");
    CUresult (*launch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **,
                       void **) = LAYER("cuLaunchKernel");
    CUresult (*as_function)(void **, void *) = LAYER("cuKernelGetFunction");
    CUresult r = attribute(&value, 0, k, 0);
    snprintf(line, sizeof line, "%s: cuKernelGetAttribute refused with 218 (got %d)", what, r);
    check(line, r == 218);
    r = launch(k, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL);
    snprintf(line, sizeof line, "%s: cuLaunchKernel of the kernel refused with 218 (got %d)", what, r);
    check(line, r == 218);
    void *function = (void *)0xdeadbeef;
    r = as_function(&function, k);
    snprintf(line, sizeof line, "%s: cuKernelGetFunction refused with 218 and a null handle (got %d, %p)", what, r,
             function);
    check(line, r == 218 && function == NULL);
}

int main(int argc, char **argv) {
    if (argc != 2 || (strcmp(argv[1], "library") && strcmp(argv[1], "module") && strcmp(argv[1], "contexts"))) {
        fprintf(stderr, "usage: test_guard library|module|contexts\n");
        return 2;
    }
    int contexts_mode = !strcmp(argv[1], "contexts");
    library_mode = contexts_mode || !strcmp(argv[1], "library");
    const char *path = getenv("DRVRUN_LIBCUDA");
    cuda = dlopen(path ? path : "libcuda.so.1", RTLD_NOW);
    if (!cuda) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return 2;
    }
    CUresult (*init)(unsigned) = LAYER("cuInit");
    CUresult (*retain)(void **, int) = LAYER("cuDevicePrimaryCtxRetain");
    CUresult (*set_current)(void *) = LAYER("cuCtxSetCurrent");
    CUresult (*allocate)(CUdeviceptr *, size_t) = LAYER("cuMemAlloc_v2");
    void *context = NULL;
    if (init(0) || retain(&context, 0) || set_current(context) || allocate(&word, sizeof(unsigned))) {
        fprintf(stderr, "no CUDA context\n");
        return 2;
    }
    static char proxy_real[PATH_MAX];
    proxy_path = realpath(path ? path : "libcuda.so.1", proxy_real) ? proxy_real : "";
    dl_iterate_phdr(find_nvidia, NULL);
    nvidia = nvidia_path[0] ? dlopen(nvidia_path, RTLD_NOW | RTLD_NOLOAD) : NULL;
    if (!nvidia) {
        fprintf(stderr, "NVIDIA's libcuda.so is not loaded beside %s\n", proxy_path);
        return 2;
    }
    printf("through the %s interface; NVIDIA's driver: %s\n", argv[1], nvidia_path);

    CUresult r;
    if (library_mode) {
        CUresult (*load)(void **, const void *, int *, void **, unsigned, int *, void **, unsigned) =
            LAYER("cuLibraryLoadData");
        r = load(&loaded, module_text, NULL, NULL, 0, NULL, NULL, 0);
    } else {
        CUresult (*load)(void **, const void *) = LAYER("cuModuleLoadData");
        r = load(&loaded, module_text);
    }
    check("the module loads", r == 0);
    if (r) return 1;
    // (Not before the contexts mode's questions: through the layer, add_one makes the layer know the
    // library's module in this context, which the questions are to find unknown.)
    if (!contexts_mode) check("add_one(10) is 11", add_one(10) == 11);
    void *good = NULL, *refused = (void *)0xdeadbeef;
    CUresult (*get)(void **, void *, const char *) = LAYER(library_mode ? "cuLibraryGetKernel" : "cuModuleGetFunction");
    r = get(&refused, loaded, "refused_by_layer");
    check("the layer refuses refused_by_layer with 218 and a null handle", r == 218 && refused == NULL);

    CUresult (*nvidia_function)(void **, void *, const char *) = NVIDIA("cuModuleGetFunction");
    if (contexts_mode) {
        CUresult (*nvidia_kernel)(void **, void *, const char *) = NVIDIA("cuLibraryGetKernel");
        CUresult (*nvidia_kernel_function)(void **, void *) = NVIDIA("cuKernelGetFunction");
        CUresult (*attribute)(int *, int, void *) = LAYER("cuFuncGetAttribute");
        CUresult (*occupancy)(int *, void *, int, size_t) = LAYER("cuOccupancyMaxActiveBlocksPerMultiprocessor");
        CUresult (*create)(void **, unsigned, int) = LAYER("cuCtxCreate_v2");
        CUresult (*destroy)(void *) = LAYER("cuCtxDestroy_v2");
        void *kernel = NULL, *function = NULL, *other = NULL;
        int value = 0;
        char line[200];
        check("NVIDIA's cuLibraryGetKernel gives the refused kernel (VF_LAZY=0)",
              nvidia_kernel(&kernel, loaded, "refused_by_layer") == 0 && kernel);
        check("NVIDIA's cuKernelGetFunction gives its function in context A",
              kernel && nvidia_kernel_function(&function, kernel) == 0 && function);
        for (int elsewhere = 0; elsewhere < 2 && function; ++elsewhere) {
            if (elsewhere) create(&other, 0, 0);
            else set_current(NULL);
            r = attribute(&value, 0, function);
            printf("  asked about while %s: %d\n", elsewhere ? "context B is current" : "no context is current", r);
            set_current(context);
            r = attribute(&value, 0, function);
            snprintf(line, sizeof line, "then in context A: cuFuncGetAttribute refused with 218 (got %d)", r);
            check(line, r == 218);
            r = occupancy(&value, function, 32, 0);
            snprintf(line, sizeof line, "then in context A: cuOccupancyMaxActiveBlocksPerMultiprocessor refused with 218 (got %d)",
                     r);
            check(line, r == 218);
        }
        if (other) destroy(other);
        set_current(context);
    } else if (library_mode) {
        CUresult (*nvidia_kernel)(void **, void *, const char *) = NVIDIA("cuLibraryGetKernel");
        CUresult (*nvidia_kernel_function)(void **, void *) = NVIDIA("cuKernelGetFunction");
        void *kernel = NULL;
        r = nvidia_kernel(&kernel, loaded, "refused_by_layer");
        printf("  NVIDIA's cuLibraryGetKernel(refused_by_layer) -> %d %p\n", r, kernel);
        if (r == 0 && kernel) {
            refused_kernel("the library's kernel from NVIDIA's driver", kernel);
            /* Its function in this context, before the program asks for the library's module. */
            void *function = NULL;
            check("NVIDIA's cuKernelGetFunction gives its function", nvidia_kernel_function(&function, kernel) == 0);
            CUresult (*get_good)(void **, void *, const char *) = NVIDIA("cuLibraryGetKernel");
            void *good_kernel = NULL;
            get_good(&good_kernel, loaded, "add_one");
            nvidia_kernel_function(&good, good_kernel);
            refused_function("the library kernel's function from NVIDIA's driver", function, good);
        } else {
            printf("  (a deferred library: the kernel is in no module of the driver until it is loaded whole)\n");
        }
        CUresult (*library_module)(void **, void *) = LAYER("cuLibraryGetModule");
        void *module = NULL;
        check("the library's module", library_module(&module, loaded) == 0);
        void *function = NULL;
        check("NVIDIA's cuModuleGetFunction gives the function of the library's module",
              nvidia_function(&function, module, "refused_by_layer") == 0 && function);
        nvidia_function(&good, module, "add_one");
        refused_function("the function of the library's module from NVIDIA's driver", function, good);
    } else {
        void *function = NULL;
        check("NVIDIA's cuModuleGetFunction gives the function",
              nvidia_function(&function, loaded, "refused_by_layer") == 0 && function);
        nvidia_function(&good, loaded, "add_one");
        refused_function("the function from NVIDIA's driver", function, good);
    }
    /* The context is still usable: nothing trapped. */
    check("add_one(41) is 42 afterwards", add_one(41) == 42);

    /* What NVIDIA's driver answers to a launch of a function that is not one, for the record. */
    CUresult (*nvidia_launch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *,
                              void **, void **) = NVIDIA("cuLaunchKernel");
    printf("  NVIDIA's cuLaunchKernel with a null function -> %d\n",
           nvidia_launch(NULL, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL));
    printf("%s\n", failed ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED");
    return failed;
}
