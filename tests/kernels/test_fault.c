/*
 * test_fault.c: a kernel that faults on purpose, launched after kernels that run, for what the layer
 * says after a fault (`fault.rs` of the proxy). THE MODES THAT FAULT STOP THE GPU'S CONTEXT ON
 * PURPOSE: each adds lines with "Xid" to the kernel's log. They are not run by tests/gpu/run.sh
 * unless VF_TEST_FAULTS=1.
 *
 *   test_fault probe-null|probe-trap|probe-misaligned
 *       (faults) the kernel writes through a null pointer, executes `trap`, or reads a word at an odd
 *       address; then every call below is made once and what it returns is printed: what NVIDIA's
 *       driver answers after a fault, with DRVRUN_LIBCUDA naming NVIDIA's libcuda.so.1 itself.
 *   test_fault fault
 *       (faults) add_one twice with a synchronisation after each, add_one once more, write_null,
 *       add_one again, then cuCtxSynchronize; the layer is to name write_null among the kernels
 *       launched since the last successful synchronisation, and none of the first two add_one.
 *   test_fault fault-graph
 *       (faults) the same kernels captured in a graph on a stream of its own, the graph launched,
 *       then cuStreamSynchronize; then what cuStreamGetCtx answers for that stream.
 *   test_fault fault-copy
 *       (faults) the fault is first returned by cuMemcpyDtoH (a call that the layer does not route),
 *       then the program destroys its context.
 *   test_fault fault-exit
 *       (faults) the same, and the program exits without destroying its context. (A second context
 *       in the process of a fault cannot be made: on a V100 with driver 580.178.04, cuCtxCreate or
 *       the first call in it returned 700, 2026-11-24.)
 *   test_fault nofault
 *       the sequence of `fault` with add_one in place of write_null: nothing faults.
 *   test_fault rate N
 *       launches of an empty kernel, N at a time, five times: launches per second.
 *
 * The kernels are PTX for sm_75 (the layer retargets them), or for sm_70 when VF_FAULT_PTX70 is set
 * (for NVIDIA's driver alone, which does not compile PTX for a later GPU).
 *
 * Build: gcc -O2 -Wall -o test_fault test_fault.c -ldl
 * Run:   DRVRUN_LIBCUDA=<proxy> CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> ./test_fault fault
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef int CUresult;
typedef unsigned long long CUdeviceptr;

static const char module_body[] =
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
    /* Writes through the pointer it is given, which the program gives null. */
    ".visible .entry write_null(.param .u64 p)\n"
    "{\n"
    "    .reg .u64 %rd<1>;\n"
    "    .reg .u32 %r<1>;\n"
    "    ld.param.u64 %rd0, [p];\n"
    "    mov.u32 %r0, 7;\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n"
    ".visible .entry trap_now(.param .u64 p)\n"
    "{\n"
    "    trap;\n"
    "    ret;\n"
    "}\n"
    /* Reads the word one byte after the one it is given. */
    ".visible .entry misaligned(.param .u64 p)\n"
    "{\n"
    "    .reg .u64 %rd<3>;\n"
    "    .reg .u32 %r<1>;\n"
    "    ld.param.u64 %rd0, [p];\n"
    "    cvta.to.global.u64 %rd1, %rd0;\n"
    "    add.u64 %rd2, %rd1, 1;\n"
    "    ld.global.u32 %r0, [%rd2];\n"
    "    st.global.u32 [%rd1], %r0;\n"
    "    ret;\n"
    "}\n"
    ".visible .entry nop()\n"
    "{\n"
    "    ret;\n"
    "}\n";

static void *cuda;
#define DRV(name) dlsym(cuda, name)
typedef CUresult (*launch_t)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *,
                             void **, void **);
static launch_t launch;
static CUresult (*synchronize)(void);
static CUresult (*stream_synchronize)(void *);
static CUresult (*from_device)(void *, CUdeviceptr, size_t);
static CUresult (*to_device)(CUdeviceptr, const void *, size_t);
static void *module, *library, *context;
static CUdeviceptr word;
static const char *module_text;

/* The functions of the module, taken before anything faults. */
static const char *const names[] = {"add_one", "write_null", "trap_now", "misaligned", "nop"};
static void *functions[5];

static void *function(const char *name) {
    for (int i = 0; i < 5; i++)
        if (!strcmp(names[i], name)) return functions[i];
    exit(2);
}

/* Launch `name` with one block of 32 threads on `stream`, its argument `pointer`. */
static CUresult run(const char *name, CUdeviceptr pointer, void *stream) {
    void *args[] = {&pointer};
    CUresult r = launch(function(name), 1, 1, 1, 32, 1, 1, 0, stream, args, NULL);
    printf("launch %s -> %d\n", name, r);
    return r;
}

static void say(const char *call, CUresult r) { printf("after the fault: %-40s -> %d\n", call, r); }

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* What the driver answers after a fault, call by call (the first ones in the order a program meets
   them). */
static int probe(const char *kind) {
    const char *name = !strcmp(kind, "null") ? "write_null" : !strcmp(kind, "trap") ? "trap_now" : "misaligned";
    CUresult (*load_library)(void **, const void *, int *, void **, unsigned, int *, void **, unsigned) =
        DRV("cuLibraryLoadData");
    CUresult (*get_kernel)(void **, void *, const char *) = DRV("cuLibraryGetKernel");
    CUresult (*func_name)(const char **, void *) = DRV("cuFuncGetName");
    CUresult (*kernel_name)(const char **, void *) = DRV("cuKernelGetName");
    CUresult (*func_module)(void **, void *) = DRV("cuFuncGetModule");
    CUresult (*kernel_library)(void **, void *) = DRV("cuKernelGetLibrary");
    CUresult (*get_current)(void **) = DRV("cuCtxGetCurrent");
    CUresult (*stream_query)(void *) = DRV("cuStreamQuery");
    CUresult (*event_create)(void **, unsigned) = DRV("cuEventCreate");
    CUresult (*event_record)(void *, void *) = DRV("cuEventRecord");
    CUresult (*event_query)(void *) = DRV("cuEventQuery");
    CUresult (*event_synchronize)(void *) = DRV("cuEventSynchronize");
    CUresult (*is_capturing)(void *, int *) = DRV("cuStreamIsCapturing");
    CUresult (*load)(void **, const void *) = DRV("cuModuleLoadData");
    CUresult (*unload)(void *) = DRV("cuModuleUnload");
    CUresult (*library_unload)(void *) = DRV("cuLibraryUnload");
    CUresult (*push)(void *) = DRV("cuCtxPushCurrent_v2");
    CUresult (*pop)(void **) = DRV("cuCtxPopCurrent_v2");
    CUresult (*attribute)(int *, int, void *) = DRV("cuFuncGetAttribute");
    void *kernel = NULL, *event = NULL, *other = NULL, *owner = NULL;
    load_library(&library, module_text, NULL, NULL, 0, NULL, NULL, 0);
    get_kernel(&kernel, library, "add_one");
    event_create(&event, 0);
    void *add = function("add_one");
    run("add_one", word, NULL);
    printf("synchronize -> %d\n", synchronize());
    double start = now();
    run(name, !strcmp(kind, "null") ? 0 : word, NULL);
    CUresult r = run("add_one", word, NULL);
    printf("the launch right after the faulting one -> %d (%.0f us after it)\n", r, (now() - start) * 1e6);
    event_record(event, NULL);
    usleep(500000);
    const char *text = NULL;
    void *current = NULL;
    int status = 0, value = 0;
    unsigned host = 0;
    say("cuStreamQuery(0)", stream_query(NULL));
    say("cuEventQuery", event_query(event));
    say("cuEventSynchronize", event_synchronize(event));
    say("cuLaunchKernel(add_one)", run("add_one", word, NULL));
    r = func_name(&text, add);
    printf("after the fault: %-40s -> %d (%s)\n", "cuFuncGetName(add_one)", r, r ? "-" : text);
    r = kernel_name(&text, kernel);
    printf("after the fault: %-40s -> %d (%s)\n", "cuKernelGetName(add_one)", r, r ? "-" : text);
    say("cuFuncGetModule", func_module(&owner, add));
    say("cuKernelGetLibrary", kernel_library(&owner, kernel));
    say("cuFuncGetAttribute", attribute(&value, 4, add));
    r = get_current(&current);
    printf("after the fault: %-40s -> %d (%s)\n", "cuCtxGetCurrent", r, current == context ? "the context" : "another");
    say("cuStreamIsCapturing(0)", is_capturing(NULL, &status));
    say("cuMemcpyDtoH", from_device(&host, word, sizeof host));
    say("cuCtxSynchronize", synchronize());
    say("cuStreamSynchronize(0)", stream_synchronize(NULL));
    say("cuModuleLoadData", load(&other, module_text));
    say("cuCtxPushCurrent", push(context));
    say("cuCtxPopCurrent", pop(&current));
    say("cuModuleUnload", unload(module));
    say("cuLibraryUnload", library_unload(library));
    if (!strcmp(kind, "misaligned")) {
        CUresult (*destroy)(void *) = DRV("cuCtxDestroy_v2");
        say("cuCtxDestroy", destroy(context));
    } else {
        CUresult (*release)(int) = DRV("cuDevicePrimaryCtxRelease_v2");
        say("cuDevicePrimaryCtxRelease", release(0));
    }
    return 0;
}

/* Graph capture: the kernels of `fault` on a stream of its own, launched as a graph. */
static int graph(void) {
    CUresult (*stream_create)(void **, unsigned) = DRV("cuStreamCreate");
    CUresult (*begin)(void *, int) = DRV("cuStreamBeginCapture_v2");
    CUresult (*end)(void *, void **) = DRV("cuStreamEndCapture");
    CUresult (*instantiate)(void **, void *, unsigned long long) = DRV("cuGraphInstantiateWithFlags");
    CUresult (*graph_launch)(void *, void *) = DRV("cuGraphLaunch");
    void *stream = NULL, *captured = NULL, *exec = NULL;
    stream_create(&stream, 1);
    run("add_one", word, stream);
    printf("stream synchronize -> %d\n", stream_synchronize(stream));
    printf("begin capture -> %d\n", begin(stream, 2 /* relaxed */));
    run("add_one", word, stream);
    run("write_null", 0, stream);
    run("add_one", word, stream);
    printf("end capture -> %d\n", end(stream, &captured));
    printf("instantiate -> %d\n", instantiate(&exec, captured, 0));
    printf("graph launch -> %d\n", graph_launch(exec, stream));
    CUresult r = stream_synchronize(stream);
    printf("stream synchronize -> %d\n", r);
    /* What the layer asks to know the context of a stream whose call met a fault. */
    CUresult (*stream_context)(void *, void **) = DRV("cuStreamGetCtx");
    void *of = NULL;
    CUresult c = stream_context(stream, &of);
    printf("after the fault: cuStreamGetCtx -> %d (%s)\n", c, c ? "-" : of == context ? "the context" : "another");
    return r ? 0 : 1;
}

static int rate(long count) {
    void *nop = function("nop");
    void *args[] = {NULL};
    for (int round = 0; round < 6; round++) {
        double start = now();
        for (long i = 0; i < count; i++) {
            CUresult r = launch(nop, 1, 1, 1, 32, 1, 1, 0, NULL, args, NULL);
            if (r) {
                printf("launch -> %d\n", r);
                return 1;
            }
        }
        double launched = now() - start;
        CUresult r = synchronize();
        double done = now() - start;
        /* (The first round compiles and warms up: not counted.) */
        if (round) printf("rate: %ld launches in %.4f s: %.0f launches/s (%.0f with the wait for the GPU) %d\n",
                          count, launched, count / launched, count / done, r);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: test_fault probe-null|probe-trap|probe-misaligned|fault|fault-graph|fault-copy|fault-exit|nofault|rate N\n");
        return 2;
    }
    const char *mode = argv[1];
    const char *path = getenv("DRVRUN_LIBCUDA");
    cuda = dlopen(path ? path : "libcuda.so.1", RTLD_NOW);
    if (!cuda) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return 2;
    }
    static char text[4096];
    snprintf(text, sizeof text, ".version 7.0\n.target %s\n%s", getenv("VF_FAULT_PTX70") ? "sm_70" : "sm_75",
             module_body);
    module_text = text;
    CUresult (*init)(unsigned) = DRV("cuInit");
    CUresult (*retain)(void **, int) = DRV("cuDevicePrimaryCtxRetain");
    CUresult (*create)(void **, unsigned, int) = DRV("cuCtxCreate_v2");
    CUresult (*set_current)(void *) = DRV("cuCtxSetCurrent");
    CUresult (*allocate)(CUdeviceptr *, size_t) = DRV("cuMemAlloc_v2");
    CUresult (*load)(void **, const void *) = DRV("cuModuleLoadData");
    launch = (launch_t)DRV("cuLaunchKernel");
    synchronize = DRV("cuCtxSynchronize");
    stream_synchronize = DRV("cuStreamSynchronize");
    from_device = DRV("cuMemcpyDtoH_v2");
    to_device = DRV("cuMemcpyHtoD_v2");
    int own_context = !strcmp(mode, "probe-misaligned") || !strcmp(mode, "fault-copy") || !strcmp(mode, "fault-exit");
    CUresult r = init(0);
    if (!r) r = own_context ? create(&context, 0, 0) : retain(&context, 0);
    if (!r && !own_context) r = set_current(context);
    if (!r) r = allocate(&word, sizeof(unsigned));
    if (!r) r = load(&module, module_text);
    CUresult (*get)(void **, void *, const char *) = DRV("cuModuleGetFunction");
    for (int i = 0; i < 5 && !r; i++) r = get(&functions[i], module, names[i]);
    if (r) {
        fprintf(stderr, "no CUDA context, module or function (%d)\n", r);
        return 2;
    }
    unsigned zero = 0, value = 0;
    to_device(word, &zero, sizeof zero);
    if (!strncmp(mode, "probe-", 6)) return probe(mode + 6);
    if (!strcmp(mode, "rate") && argc == 3) return rate(atol(argv[2]));
    if (!strcmp(mode, "fault-graph")) return graph();
    int faults = strcmp(mode, "nofault") != 0;
    if (!faults || !strcmp(mode, "fault") || !strcmp(mode, "fault-copy") || !strcmp(mode, "fault-exit")) {
        run("add_one", word, NULL);
        printf("synchronize -> %d\n", synchronize());
        run("add_one", word, NULL);
        printf("stream synchronize -> %d\n", stream_synchronize(NULL));
        run("add_one", word, NULL);
        if (faults) run("write_null", 0, NULL);
        else run("add_one", word, NULL);
        run("add_one", word, NULL);
        if (!strcmp(mode, "fault-copy") || !strcmp(mode, "fault-exit")) {
            usleep(200000);
            r = from_device(&value, word, sizeof value);
            printf("copy -> %d\n", r);
            if (!strcmp(mode, "fault-copy")) {
                CUresult (*destroy)(void *) = DRV("cuCtxDestroy_v2");
                printf("context destroyed -> %d\n", destroy(context));
            }
            return r ? 0 : 1;
        }
        r = synchronize();
        printf("synchronize -> %d\n", r);
        from_device(&value, word, sizeof value);
        if (!faults) {
            printf("word %u (5 expected)\n%s\n", value, value == 5 && !r ? "NO FAULT" : "WRONG");
            return value == 5 && !r ? 0 : 1;
        }
        return r ? 0 : 1;
    }
    fprintf(stderr, "unknown mode %s\n", mode);
    return 2;
}
