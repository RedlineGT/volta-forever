/*
 * test_reported.c: instructions that need a multicast address (multimem.*), which no GPU of a
 * process of V100s can have, are replaced by a reported trap, and the kernels that hold them, or
 * may reach them, run right on every path that does not reach one. A form that a program could
 * really execute keeps its kernels refused.
 *
 *   test_reported library|module    the cases below, through cuLibraryLoadData + cuLibraryGetKernel
 *                                   or cuModuleLoadData + cuModuleGetFunction
 *   test_reported multicast         what the driver says of multicast on each GPU of the process:
 *                                   CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, cuMulticastGetGranularity
 *                                   and cuMulticastCreate (no layer needed: run it both ways)
 *   test_reported fault             launches `pick` so that its call takes the function with the
 *                                   multimem instruction: the kernel stops at the reported trap and
 *                                   the context ends (the GPU writes lines with "Xid" in the kernel's
 *                                   log). Only for tests/gpu/run.sh with VF_TEST_FAULTS=1.
 *
 * The module (PTX for sm_90, which the layer adapts) has, as NCCL has:
 *   - pick: a call through a pointer taken from a table (`table`) of two device functions of one
 *     prototype; the program's argument chooses which. `twice` doubles a word, `gathered` reads it
 *     with multimem.ld_reduce. Until 0.1.4 the call that may reach `gathered` refused `pick`;
 *   - polled: multimem.ld_reduce behind a test of a flag that the program passes, as NCCL's step
 *     counters are read (0: a plain load);
 *   - stored: a kernel whose only work is multimem.st, given (never launched here);
 *   - plain: nothing of later GPUs (a control of the module).
 * The second module has the same `pick`, with a function that holds barrier.cluster.arrive, an
 * instruction of later GPUs that a program can really execute: `pick` stays refused (218).
 * With VF_REPORTED_TRAPS=0 the first module's pick, polled and stored are refused too (the run.sh
 * case says which is expected: argument 2, `refused`).
 *
 * Build: gcc -O2 -Wall -o test_reported test_reported.c -ldl
 * Run:   DRVRUN_LIBCUDA=<proxy> CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> ./test_reported library
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef unsigned long long CUdeviceptr;

#define HEADER ".version 8.1\n.target sm_90\n.address_size 64\n\n"
#define PROTOTYPE "    prototype: .callprototype (.param .b32 _) _ (.param .b64 _);\n"
/* pick(out, which, address): out = table[which](address). */
#define PICK                                                                  \
    ".visible .entry pick(.param .u64 out, .param .u32 which, .param .u64 at)\n" \
    "{\n"                                                                     \
    "    .reg .u64 %rd<6>;\n"                                                 \
    "    .reg .u32 %r<3>;\n"                                                  \
    "    ld.param.u64 %rd1, [out];\n"                                         \
    "    ld.param.u32 %r1, [which];\n"                                        \
    "    ld.param.u64 %rd2, [at];\n"                                          \
    "    mov.u64 %rd3, table;\n"                                              \
    "    mul.wide.u32 %rd4, %r1, 8;\n"                                        \
    "    add.u64 %rd3, %rd3, %rd4;\n"                                         \
    "    ld.global.u64 %rd5, [%rd3];\n"                                       \
    "    {\n"                                                                 \
    "    .param .b64 a;\n"                                                    \
    "    .param .b32 r;\n"                                                    \
    "    st.param.b64 [a], %rd2;\n"                                           \
    PROTOTYPE                                                                 \
    "    call (r), %rd5, (a), prototype;\n"                                   \
    "    ld.param.b32 %r2, [r];\n"                                            \
    "    }\n"                                                                 \
    "    cvta.to.global.u64 %rd1, %rd1;\n"                                    \
    "    st.global.u32 [%rd1], %r2;\n"                                        \
    "    ret;\n"                                                              \
    "}\n"
#define TWICE                                                                 \
    ".func (.param .b32 r) twice(.param .b64 a)\n"                            \
    "{\n"                                                                     \
    "    .reg .u64 %rd<2>;\n"                                                 \
    "    .reg .u32 %r<3>;\n"                                                  \
    "    ld.param.u64 %rd1, [a];\n"                                           \
    "    cvta.to.global.u64 %rd1, %rd1;\n"                                    \
    "    ld.global.u32 %r1, [%rd1];\n"                                        \
    "    add.u32 %r2, %r1, %r1;\n"                                            \
    "    st.param.b32 [r], %r2;\n"                                            \
    "    ret;\n"                                                              \
    "}\n"

static const char module_reported[] =
    HEADER
    ".func (.param .b32 r) twice(.param .b64 a);\n"
    ".func (.param .b32 r) gathered(.param .b64 a);\n"
    ".global .align 8 .u64 table[2] = {twice, gathered};\n"
    "\n"
    TWICE
    ".func (.param .b32 r) gathered(.param .b64 a)\n"
    "{\n"
    "    .reg .u64 %rd<2>;\n"
    "    .reg .u32 %r<2>;\n"
    "    ld.param.u64 %rd1, [a];\n"
    "    cvta.to.global.u64 %rd1, %rd1;\n"
    "    multimem.ld_reduce.relaxed.sys.global.add.u32 %r1, [%rd1];\n"
    "    st.param.b32 [r], %r1;\n"
    "    ret;\n"
    "}\n"
    PICK
    ".visible .entry polled(.param .u64 out, .param .u32 flag)\n"
    "{\n"
    "    .reg .u64 %rd<3>;\n"
    "    .reg .u32 %r<3>;\n"
    "    .reg .pred %p<2>;\n"
    "    ld.param.u64 %rd1, [out];\n"
    "    ld.param.u32 %r1, [flag];\n"
    "    cvta.to.global.u64 %rd2, %rd1;\n"
    "    setp.ne.u32 %p1, %r1, 0;\n"
    "    @%p1 bra MULTICAST;\n"
    "    ld.volatile.global.u32 %r2, [%rd2];\n"
    "    bra.uni DONE;\n"
    "MULTICAST:\n"
    "    multimem.ld_reduce.acquire.sys.global.min.u32 %r2, [%rd2];\n"
    "DONE:\n"
    "    add.u32 %r2, %r2, 1000;\n"
    "    st.global.u32 [%rd2], %r2;\n"
    "    ret;\n"
    "}\n"
    ".visible .entry stored(.param .u64 out, .param .u32 value)\n"
    "{\n"
    "    .reg .u64 %rd<2>;\n"
    "    .reg .u32 %r<2>;\n"
    "    ld.param.u64 %rd1, [out];\n"
    "    ld.param.u32 %r1, [value];\n"
    "    multimem.st.global.b32 [%rd1], %r1;\n"
    "    ret;\n"
    "}\n"
    ".visible .entry plain(.param .u64 out)\n"
    "{\n"
    "    .reg .u64 %rd<2>;\n"
    "    .reg .u32 %r<2>;\n"
    "    ld.param.u64 %rd1, [out];\n"
    "    cvta.to.global.u64 %rd1, %rd1;\n"
    "    ld.global.u32 %r1, [%rd1];\n"
    "    add.u32 %r1, %r1, 7;\n"
    "    st.global.u32 [%rd1], %r1;\n"
    "    ret;\n"
    "}\n";

static const char module_refused[] =
    HEADER
    ".func (.param .b32 r) twice(.param .b64 a);\n"
    ".func (.param .b32 r) clustered(.param .b64 a);\n"
    ".global .align 8 .u64 table[2] = {twice, clustered};\n"
    "\n"
    TWICE
    ".func (.param .b32 r) clustered(.param .b64 a)\n"
    "{\n"
    "    .reg .u32 %r<2>;\n"
    "    barrier.cluster.arrive;\n"
    "    barrier.cluster.wait;\n"
    "    mov.u32 %r1, 1;\n"
    "    st.param.b32 [r], %r1;\n"
    "    ret;\n"
    "}\n"
    PICK
    ".visible .entry plain(.param .u64 out)\n"
    "{\n"
    "    .reg .u64 %rd<2>;\n"
    "    .reg .u32 %r<2>;\n"
    "    ld.param.u64 %rd1, [out];\n"
    "    cvta.to.global.u64 %rd1, %rd1;\n"
    "    ld.global.u32 %r1, [%rd1];\n"
    "    add.u32 %r1, %r1, 7;\n"
    "    st.global.u32 [%rd1], %r1;\n"
    "    ret;\n"
    "}\n";

static void *cuda;
#define DRIVER(name) dlsym(cuda, name)
static CUresult (*launch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **,
                          void **);
static CUresult (*synchronize)(void);
static CUresult (*to_device)(CUdeviceptr, const void *, size_t);
static CUresult (*from_device)(void *, CUdeviceptr, size_t);

static int by_module, failed;
static void *loaded;
static CUdeviceptr word;

static void check(const char *what, int ok) {
    printf("  %s  %s\n", ok ? "ok    " : "WRONG ", what);
    failed |= !ok;
}

static CUresult load(const char *text) {
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
}

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

/* Launch `name` with `args` (the word first) on the word set to `in`: what it leaves there, or
 * 0xdead0000 + the error. */
static unsigned run(const char *name, unsigned in, void **args) {
    void *function = NULL;
    CUresult r = find(name, &function);
    if (r) return 0xdead0000u + r;
    unsigned out = 0;
    to_device(word, &in, sizeof in);
    r = launch(function, 1, 1, 1, 32, 1, 1, 0, NULL, args, NULL);
    if (!r) r = synchronize();
    if (r) return 0xdead0000u + r;
    from_device(&out, word, sizeof out);
    return out;
}

static void given(const char *name) {
    void *function = NULL;
    char what[160];
    CUresult r = find(name, &function);
    snprintf(what, sizeof what, "%s is given (lookup -> %d)", name, r);
    check(what, r == 0 && function != NULL);
}

static void refused(const char *name) {
    CUresult (*get)(void **, void *, const char *) = DRIVER(by_module ? "cuModuleGetFunction" : "cuLibraryGetKernel");
    void *handle = (void *)0xdeadbeef;
    char what[160];
    CUresult r = get(&handle, loaded, name);
    snprintf(what, sizeof what, "%s refused with error 218 and a null handle (got %d, %p)", name, r, handle);
    check(what, r == 218 && handle == NULL);
}

/* What the driver says of multicast for each GPU of the process. */
static int multicast(void) {
    CUresult (*count)(int *) = DRIVER("cuDeviceGetCount");
    CUresult (*attribute)(int *, int, int) = DRIVER("cuDeviceGetAttribute");
    CUresult (*granularity)(size_t *, const void *, int) = DRIVER("cuMulticastGetGranularity");
    CUresult (*create)(unsigned long long *, const void *) = DRIVER("cuMulticastCreate");
    int devices = 0;
    count(&devices);
    for (int device = 0; device < devices; device++) {
        int supported = -1;
        CUresult r = attribute(&supported, 132 /* CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED */, device);
        printf("GPU %d: CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED -> %d, value %d\n", device, r, supported);
    }
    /* CUmulticastObjectProp: numDevices, size, handleTypes, flags. */
    struct { unsigned num_devices; size_t size; unsigned long long handle_types; unsigned long long flags; } prop = {
        (unsigned)devices, 2u << 20, 0, 0};
    size_t step = 0;
    printf("cuMulticastGetGranularity -> %d (%zu)\n", granularity(&step, &prop, 0), step);
    unsigned long long handle = 0;
    printf("cuMulticastCreate -> %d\n", create(&handle, &prop));
    return 0;
}

int main(int argc, char **argv) {
    const char *mode = argc == 2 ? argv[1] : "";
    if (strcmp(mode, "library") && strcmp(mode, "module") && strcmp(mode, "multicast") && strcmp(mode, "fault") &&
        strcmp(mode, "refused")) {
        fprintf(stderr, "usage: test_reported library|module|refused|multicast|fault\n");
        return 2;
    }
    by_module = !strcmp(mode, "module");
    int switched_off = !strcmp(mode, "refused");
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
    if (!strcmp(mode, "multicast")) return multicast();

    unsigned which = 0, flag = 0, value = 5;
    void *pick_args[] = {&word, &which, &word};
    void *polled_args[] = {&word, &flag};
    void *plain_args[] = {&word};
    (void)value;
    if (!strcmp(mode, "fault")) {
        CUresult r = load(module_reported);
        printf("module loaded -> %d\n", r);
        which = 1;
        printf("pick through the multimem function -> 0x%x\n", run("pick", 21, pick_args));
        fflush(stdout);
        return 0;
    }

    printf("through the %s interface\n", by_module ? "module" : "library");
    CUresult r = load(module_reported);
    printf("module with multimem instructions: load -> %d\n", r);
    check("the module loads", r == 0);
    if (r == 0) {
        check("plain(10) is 17", run("plain", 10, plain_args) == 17);
        if (switched_off) {
            refused("pick");
            refused("polled");
            refused("stored");
        } else {
            which = 0;
            check("pick through twice: 2 x 21 = 42", run("pick", 21, pick_args) == 42);
            check("pick again: 2 x 50 = 100", run("pick", 50, pick_args) == 100);
            flag = 0;
            check("polled without the flag: 1000 + 5", run("polled", 5, polled_args) == 1005);
            given("stored");
        }
        check("plain(35) is still 42", run("plain", 35, plain_args) == 42);
        unload();
    }
    r = load(module_refused);
    printf("module with barrier.cluster.arrive behind the same call: load -> %d\n", r);
    check("the module loads", r == 0);
    if (r == 0) {
        refused("pick");
        check("plain(1) is 8", run("plain", 1, plain_args) == 8);
        unload();
    }
    unsigned back = 0, sent = 0x600d600d;
    check("the context still works", to_device(word, &sent, sizeof sent) == 0 && synchronize() == 0 &&
                                         from_device(&back, word, sizeof back) == 0 && back == sent);
    printf("%s\n", failed ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED");
    return failed;
}
