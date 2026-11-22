/*
 * test_refuse.c: kernels that cannot run on the GPU are refused when the program asks for them, and
 * the others of their module work.
 *
 *   test_refuse library    load with cuLibraryLoadData, look up with cuLibraryGetKernel
 *   test_refuse module     load with cuModuleLoadData, look up with cuModuleGetFunction
 *   test_refuse link       link with cuLinkAddData and cuLinkComplete, load what the link made with
 *                          cuModuleLoadData, look up with cuModuleGetFunction (the first module
 *                          without refused_by_compiler, which the linker would refuse whole)
 *
 * Two modules of PTX for sm_75 (so that the proxy adapts them), each with a variable its kernels
 * share. What is wrong with the kernels that must be refused:
 *   - refused_by_layer uses barrier.cluster.arrive, an instruction of sm_90 that the proxy knows it
 *     has no sequence for: it makes the function trap when it adapts the module.
 *   - refused_by_compiler and the function bad_helper use a register that is not declared, which only
 *     the driver's compiler finds: the driver refuses the whole module, and the proxy loads it again
 *     without those functions. (A register that does not exist stands for any instruction the proxy
 *     does not know: nothing it learns later makes this test pass by another way.)
 *   - caller_of_bad is sound and calls bad_helper.
 * A lookup of each of them must return CUDA_ERROR_INVALID_PTX (218) and leave NULL in the handle;
 * nothing must trap on the GPU.
 * The kernel soft has a tanh, which the proxy emulates: it must run, and be announced on the console.
 *
 * Build: gcc -O2 -Wall -o test_refuse test_refuse.c -ldl -lm
 * Run:   DRVRUN_LIBCUDA=<proxy> CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> ./test_refuse module
 */
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef unsigned long long CUdeviceptr;

#define ONE_WORD(name, body)                              \
    ".visible .entry " name "(\n"                         \
    "    .param .u64 " name "_param_0\n"                  \
    ")\n"                                                 \
    "{\n"                                                 \
    "    .reg .u64 %rd<2>;\n"                             \
    "    .reg .u32 %r<3>;\n"                              \
    "    .reg .f32 %f<2>;\n"                              \
    "    ld.param.u64 %rd0, [" name "_param_0];\n"        \
    "    cvta.to.global.u64 %rd1, %rd0;\n"                \
    body                                                  \
    "    ret;\n"                                          \
    "}\n"

static const char module_main[] =
    ".version 9.0\n"
    ".target sm_75\n"
    ".address_size 64\n"
    "\n"
    ".visible .global .align 4 .b32 counter = 5;\n"
    "\n"
    ONE_WORD("add_one", "    ld.global.u32 %r0, [%rd1];\n    add.u32 %r1, %r0, 1;\n    st.global.u32 [%rd1], %r1;\n")
    ONE_WORD("read_counter", "    ld.global.u32 %r0, [counter];\n    st.global.u32 [%rd1], %r0;\n")
    ONE_WORD("soft", "    ld.global.f32 %f0, [%rd1];\n    tanh.approx.f32 %f1, %f0;\n    st.global.f32 [%rd1], %f1;\n")
    ONE_WORD("refused_by_compiler", "    ld.global.u32 %r0, [counter];\n    mov.u32 %r1, %never_declared;\n")
    ONE_WORD("refused_by_layer", "    barrier.cluster.arrive;\n    barrier.cluster.wait;\n");

static const char module_linked[] =
    ".version 9.0\n"
    ".target sm_75\n"
    ".address_size 64\n"
    "\n"
    ".visible .global .align 4 .b32 counter = 5;\n"
    "\n"
    ONE_WORD("add_one", "    ld.global.u32 %r0, [%rd1];\n    add.u32 %r1, %r0, 1;\n    st.global.u32 [%rd1], %r1;\n")
    ONE_WORD("read_counter", "    ld.global.u32 %r0, [counter];\n    st.global.u32 [%rd1], %r0;\n")
    ONE_WORD("soft", "    ld.global.f32 %f0, [%rd1];\n    tanh.approx.f32 %f1, %f0;\n    st.global.f32 [%rd1], %f1;\n")
    ONE_WORD("refused_by_layer", "    barrier.cluster.arrive;\n    barrier.cluster.wait;\n");

static const char module_helper[] =
    ".version 9.0\n"
    ".target sm_75\n"
    ".address_size 64\n"
    "\n"
    ".visible .global .align 4 .b32 hvar = 7;\n"
    "\n"
    ".func bad_helper()\n"
    "{\n"
    "    .reg .u32 %r<2>;\n"
    "    mov.u32 %r1, %never_declared;\n"
    "    ret;\n"
    "}\n"
    ONE_WORD("caller_of_bad", "    ld.global.u32 %r0, [hvar];\n    st.global.u32 [%rd1], %r0;\n    call bad_helper;\n")
    ONE_WORD("good_reader", "    ld.global.u32 %r0, [hvar];\n    st.global.u32 [%rd1], %r0;\n");

static void *cuda;
#define DRIVER(name) dlsym(cuda, name)
static CUresult (*launch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **,
                          void **);
static CUresult (*synchronize)(void);
static CUresult (*to_device)(CUdeviceptr, const void *, size_t);
static CUresult (*from_device)(void *, CUdeviceptr, size_t);

static int by_module, by_link, failed;
static void *loaded, *link_state;
static CUdeviceptr word;

static void check(const char *what, int ok) {
    printf("  %s  %s\n", ok ? "ok    " : "WRONG ", what);
    failed |= !ok;
}

static CUresult load(const char *text) {
    if (by_link) {
        CUresult (*create)(unsigned, int *, void **, void **) = DRIVER("cuLinkCreate_v2");
        CUresult (*add)(void *, int, void *, size_t, const char *, unsigned, int *, void **) = DRIVER("cuLinkAddData_v2");
        CUresult (*complete)(void *, void **, size_t *) = DRIVER("cuLinkComplete");
        CUresult (*load_data)(void **, const void *) = DRIVER("cuModuleLoadData");
        void *image = NULL;
        size_t size = 0;
        CUresult r = create(0, NULL, NULL, &link_state);
        if (!r) r = add(link_state, 1 /* CU_JIT_INPUT_PTX */, (void *)text, strlen(text) + 1, "linked.ptx", 0, NULL, NULL);
        if (!r) r = complete(link_state, &image, &size);
        return r ? r : load_data(&loaded, image);
    }
    if (by_module) {
        CUresult (*load_data)(void **, const void *) = DRIVER("cuModuleLoadData");
        return load_data(&loaded, text);
    }
    CUresult (*load_data)(void **, const void *, int *, void **, unsigned, int *, void **, unsigned) =
        DRIVER("cuLibraryLoadData");
    return load_data(&loaded, text, NULL, NULL, 0, NULL, NULL, 0);
}

static void unload(void) {
    CUresult (*unload_it)(void *) = DRIVER(by_module ? "cuModuleUnload" : "cuLibraryUnload");
    unload_it(loaded);
    if (by_link) {
        CUresult (*destroy)(void *) = DRIVER("cuLinkDestroy");
        destroy(link_state);
    }
}

/* The function to launch for kernel `name`, or the error of the lookup. */
static CUresult find(const char *name, void **function) {
    if (by_module) {
        CUresult (*get)(void **, void *, const char *) = DRIVER("cuModuleGetFunction");
        return get(function, loaded, name);
    }
    CUresult (*get)(void **, void *, const char *) = DRIVER("cuLibraryGetKernel");
    CUresult (*as_function)(void **, void *) = DRIVER("cuKernelGetFunction");
    void *kernel = NULL;
    CUresult r = get(&kernel, loaded, name);
    return r ? r : as_function(function, kernel);
}

/* Run kernel `name` on the word `in`; what it leaves there, or 0xdead0000 + the error. */
static unsigned run(const char *name, unsigned in) {
    void *function = NULL;
    CUresult r = find(name, &function);
    if (r) return 0xdead0000u + r;
    void *args[] = {&word};
    unsigned out = 0;
    to_device(word, &in, sizeof in);
    r = launch(function, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL);
    if (!r) r = synchronize();
    if (r) return 0xdead0000u + r;
    from_device(&out, word, sizeof out);
    return out;
}

static void refused(const char *name) {
    CUresult (*get)(void **, void *, const char *) = DRIVER(by_module ? "cuModuleGetFunction" : "cuLibraryGetKernel");
    void *handle = (void *)0xdeadbeef;
    char what[160];
    CUresult r = get(&handle, loaded, name);
    snprintf(what, sizeof what, "%s refused with error 218 and a null handle (got %d, %p)", name, r, handle);
    check(what, r == 218 && handle == NULL);
}

static unsigned global(const char *name) {
    CUresult (*get)(CUdeviceptr *, size_t *, void *, const char *) =
        DRIVER(by_module ? "cuModuleGetGlobal_v2" : "cuLibraryGetGlobal");
    CUdeviceptr where = 0;
    size_t size = 0;
    unsigned value = 0xdead0000u;
    if (get(&where, &size, loaded, name) == 0 && size == sizeof value) from_device(&value, where, sizeof value);
    return value;
}

int main(int argc, char **argv) {
    if (argc != 2 || (strcmp(argv[1], "library") && strcmp(argv[1], "module") && strcmp(argv[1], "link"))) {
        fprintf(stderr, "usage: test_refuse library|module|link\n");
        return 2;
    }
    by_link = !strcmp(argv[1], "link");
    by_module = by_link || !strcmp(argv[1], "module");
    const char *path = getenv("DRVRUN_LIBCUDA");
    cuda = dlopen(path ? path : "libcuda.so.1", RTLD_NOW);
    if (!cuda) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return 2;
    }
    CUresult (*init)(unsigned) = DRIVER("cuInit");
    CUresult (*retain)(void **, int) = DRIVER("cuDevicePrimaryCtxRetain");
    CUresult (*set_current)(void *) = DRIVER("cuCtxSetCurrent");
    CUresult (*allocate)(CUdeviceptr *, size_t) = DRIVER("cuMemAlloc_v2");
    launch = DRIVER("cuLaunchKernel");
    synchronize = DRIVER("cuCtxSynchronize");
    to_device = DRIVER("cuMemcpyHtoD_v2");
    from_device = DRIVER("cuMemcpyDtoH_v2");
    void *context = NULL;
    if (init(0) || retain(&context, 0) || set_current(context) || allocate(&word, sizeof(unsigned))) {
        fprintf(stderr, "no CUDA context\n");
        return 2;
    }

    printf("through the %s interface\n", argv[1]);
    CUresult r = load(by_link ? module_linked : module_main);
    printf("module with %s: load -> %d\n",
           by_link ? "a kernel the proxy refuses" : "a kernel the compiler refuses and one the proxy refuses", r);
    check("the module loads", r == 0);
    if (r == 0) {
        check("add_one(10) is 11", run("add_one", 10) == 11);
        check("read_counter reads 5", run("read_counter", 0) == 5);
        check("the host reads 5 in the variable", global("counter") == 5);
        float half = 0.5f, got;
        unsigned bits, out;
        memcpy(&bits, &half, sizeof bits);
        out = run("soft", bits);
        memcpy(&got, &out, sizeof got);
        printf("  soft(0.5) = %.6f, tanh(0.5) = %.6f\n", got, tanhf(half));
        check("soft computes tanh", fabsf(got - tanhf(half)) < 1e-3f);
        if (!by_link) refused("refused_by_compiler");
        refused("refused_by_layer");
        check("add_one still runs", run("add_one", 41) == 42);
        unload();
    }

    r = by_link ? 1 : load(module_helper);
    if (!by_link) {
        printf("module with a function the compiler refuses: load -> %d\n", r);
        check("the module loads", r == 0);
    }
    if (r == 0) {
        check("good_reader reads 7", run("good_reader", 0) == 7);
        refused("caller_of_bad");
        check("the host reads 7 in the variable", global("hvar") == 7);
        unload();
    }

    /* The context is still usable: nothing trapped. */
    unsigned back = 0, sent = 0x600d600d;
    check("the context still works", to_device(word, &sent, sizeof sent) == 0 && synchronize() == 0 &&
                                         from_device(&back, word, sizeof back) == 0 && back == sent);
    printf("%s\n", failed ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED");
    return failed;
}
