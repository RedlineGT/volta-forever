// multi-gpu: exercise two GPUs concurrently within one process through the CUDA driver API.
// Covers concurrent module/library loading and execution on separate contexts, per-GPU concurrent
// CUDA graph capture and launch, and a cross-device saxpy -> cuMemcpyPeerAsync -> saxpy pipeline.
//
// usage: multi-gpu MODULE_FILE
// Exit 0 on pass, 77 when fewer than 2 GPUs are visible, non-zero on failure.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUlib_st* CUlibrary;
typedef struct CUkern_st* CUkernel;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;
typedef struct CUgraph_st* CUgraph;
typedef struct CUgraphExec_st* CUgraphExec;
typedef unsigned long long CUdeviceptr;

enum {
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
};

static void* lib;
#define SYM(ret, name, ...) static ret (*name)(__VA_ARGS__)
SYM(CUresult, cuInit, unsigned);
SYM(CUresult, cuDeviceGetCount, int*);
SYM(CUresult, cuDeviceGet, CUdevice*, int);
SYM(CUresult, cuDeviceGetName, char*, int, CUdevice);
SYM(CUresult, cuDeviceComputeCapability, int*, int*, CUdevice);
SYM(CUresult, cuDeviceCanAccessPeer, int*, CUdevice, CUdevice);
SYM(CUresult, cuDevicePrimaryCtxRetain, CUcontext*, CUdevice);
SYM(CUresult, cuDevicePrimaryCtxRelease_v2, CUdevice);
SYM(CUresult, cuCtxSetCurrent, CUcontext);
SYM(CUresult, cuCtxEnablePeerAccess, CUcontext, unsigned);
SYM(CUresult, cuModuleLoad, CUmodule*, const char*);
SYM(CUresult, cuModuleLoadData, CUmodule*, const void*);
SYM(CUresult, cuModuleUnload, CUmodule);
SYM(CUresult, cuModuleGetFunction, CUfunction*, CUmodule, const char*);
SYM(CUresult, cuLibraryLoadData, CUlibrary*, const void*, int*, void**, unsigned, int*, void**, unsigned);
SYM(CUresult, cuLibraryGetKernel, CUkernel*, CUlibrary, const char*);
SYM(CUresult, cuKernelGetFunction, CUfunction*, CUkernel);
SYM(CUresult, cuLibraryUnload, CUlibrary);
SYM(CUresult, cuMemAlloc_v2, CUdeviceptr*, size_t);
SYM(CUresult, cuMemFree_v2, CUdeviceptr);
SYM(CUresult, cuMemcpyHtoD_v2, CUdeviceptr, const void*, size_t);
SYM(CUresult, cuMemcpyDtoH_v2, void*, CUdeviceptr, size_t);
SYM(CUresult, cuMemcpyPeerAsync, CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream);
SYM(CUresult, cuStreamCreate, CUstream*, unsigned);
SYM(CUresult, cuStreamDestroy_v2, CUstream);
SYM(CUresult, cuStreamSynchronize, CUstream);
SYM(CUresult, cuStreamBeginCapture_v2, CUstream, unsigned);
SYM(CUresult, cuStreamEndCapture, CUstream, CUgraph*);
SYM(CUresult, cuGraphInstantiateWithFlags, CUgraphExec*, CUgraph, unsigned long long);
SYM(CUresult, cuGraphLaunch, CUgraphExec, CUstream);
SYM(CUresult, cuGraphExecDestroy, CUgraphExec);
SYM(CUresult, cuGraphDestroy, CUgraph);
SYM(CUresult, cuLaunchKernel, CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
    CUstream, void**, void**);
SYM(CUresult, cuGetErrorName, CUresult, const char**);

static void* sym(const char* name) {
    void* p = dlsym(lib, name);
    if (!p) {
        fprintf(stderr, "missing symbol %s\n", name);
        exit(2);
    }
    return p;
}

static const char* ename(CUresult r) {
    const char* s = "?";
    cuGetErrorName(r, &s);
    return s ? s : "?";
}

#define CK(call)                                                               \
    do {                                                                       \
        CUresult r_ = (call);                                                  \
        if (r_) {                                                              \
            fprintf(stderr, "FAIL %s -> %d %s\n", #call, r_, ename(r_));       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define N (1u << 18)
#define BLOCK 256u
#define GRID (N / BLOCK)

typedef struct {
    int ordinal;
    const void* image;
    float alpha;
    unsigned y_seed;
    int repeats;
    pthread_barrier_t* bar;
    long direct_bad;
    long graph_bad;
    long library_bad;
} WorkerArgs;

static void check_array(const char* label, int dev, const float* got, const float* expected, size_t n, long* out_bad) {
    long bad = 0;
    for (size_t i = 0; i < n; i++) {
        if (got[i] != expected[i]) {
            if (bad == 0) {
                fprintf(stderr, "  dev %d %s first mismatch at %zu: got %.6g want %.6g\n", dev, label, i, got[i],
                        expected[i]);
            }
            bad++;
        }
    }
    *out_bad = bad;
}

static void* concurrent_worker(void* raw) {
    WorkerArgs* w = (WorkerArgs*)raw;
    const size_t bytes = (size_t)N * sizeof(float);
    float* hx = malloc(bytes);
    float* hy_init = malloc(bytes);
    float* hy_out = malloc(bytes);
    float* expected = malloc(bytes);
    if (!hx || !hy_init || !hy_out || !expected) {
        fprintf(stderr, "oom on dev %d\n", w->ordinal);
        exit(2);
    }
    for (size_t i = 0; i < N; i++) {
        hx[i] = (float)((i + (size_t)w->ordinal * 7u) & 255u);
        hy_init[i] = (float)(((i * 3u) + w->y_seed) & 255u);
    }

    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction saxpy;
    CUstream stream;
    CUdeviceptr dx, dy;
    CK(cuDeviceGet(&dev, w->ordinal));

    // Synchronize both threads right before context retain & module load to exercise concurrent adaptation/locking.
    pthread_barrier_wait(w->bar);
    CK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CK(cuCtxSetCurrent(ctx));
    CK(cuModuleLoadData(&mod, w->image));
    CK(cuModuleGetFunction(&saxpy, mod, "saxpy"));
    CK(cuStreamCreate(&stream, 0));

    CK(cuMemAlloc_v2(&dx, bytes));
    CK(cuMemAlloc_v2(&dy, bytes));
    CK(cuMemcpyHtoD_v2(dx, hx, bytes));
    CK(cuMemcpyHtoD_v2(dy, hy_init, bytes));

    int n = (int)N;
    float a = w->alpha;
    void* args[] = {&a, &dx, &dy, &n};

    pthread_barrier_wait(w->bar);
    for (int rep = 0; rep < w->repeats; rep++) {
        CK(cuLaunchKernel(saxpy, GRID, 1, 1, BLOCK, 1, 1, 0, stream, args, NULL));
    }
    CK(cuStreamSynchronize(stream));
    CK(cuMemcpyDtoH_v2(hy_out, dy, bytes));

    float total_a = (float)w->repeats * w->alpha;
    for (size_t i = 0; i < N; i++) expected[i] = total_a * hx[i] + hy_init[i];
    check_array("concurrent-saxpy", w->ordinal, hy_out, expected, N, &w->direct_bad);

    // Concurrent thread-local stream capture + simultaneous graph replay on both GPUs.
    CK(cuMemcpyHtoD_v2(dy, hy_init, bytes));
    CK(cuCtxSetCurrent(ctx));
    pthread_barrier_wait(w->bar);

    CUgraph graph;
    CUgraphExec exec;
    CK(cuStreamBeginCapture_v2(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
    CK(cuLaunchKernel(saxpy, GRID, 1, 1, BLOCK, 1, 1, 0, stream, args, NULL));
    CK(cuLaunchKernel(saxpy, GRID, 1, 1, BLOCK, 1, 1, 0, stream, args, NULL));
    CK(cuStreamEndCapture(stream, &graph));
    CK(cuGraphInstantiateWithFlags(&exec, graph, 0));

    pthread_barrier_wait(w->bar);
    CK(cuGraphLaunch(exec, stream));
    CK(cuStreamSynchronize(stream));
    CK(cuMemcpyDtoH_v2(hy_out, dy, bytes));

    for (size_t i = 0; i < N; i++) expected[i] = (2.0f * w->alpha) * hx[i] + hy_init[i];
    check_array("concurrent-graph", w->ordinal, hy_out, expected, N, &w->graph_bad);

    CK(cuGraphExecDestroy(exec));
    CK(cuGraphDestroy(graph));
    CK(cuModuleUnload(mod));

    // Concurrent cuLibraryLoadData + deferred cuLibraryGetKernel across both threads.
    CK(cuMemcpyHtoD_v2(dy, hy_init, bytes));
    CK(cuCtxSetCurrent(ctx));
    pthread_barrier_wait(w->bar);

    CUlibrary cu_lib = NULL;
    CUkernel kern = NULL;
    CUfunction lib_saxpy = NULL;
    CK(cuLibraryLoadData(&cu_lib, w->image, NULL, NULL, 0, NULL, NULL, 0));
    CK(cuLibraryGetKernel(&kern, cu_lib, "saxpy"));
    CK(cuKernelGetFunction(&lib_saxpy, kern));

    pthread_barrier_wait(w->bar);
    for (int rep = 0; rep < w->repeats; rep++) {
        CK(cuLaunchKernel(lib_saxpy, GRID, 1, 1, BLOCK, 1, 1, 0, stream, args, NULL));
    }
    CK(cuStreamSynchronize(stream));
    CK(cuMemcpyDtoH_v2(hy_out, dy, bytes));

    for (size_t i = 0; i < N; i++) expected[i] = total_a * hx[i] + hy_init[i];
    check_array("concurrent-library", w->ordinal, hy_out, expected, N, &w->library_bad);

    CK(cuLibraryUnload(cu_lib));
    CK(cuStreamDestroy_v2(stream));
    CK(cuMemFree_v2(dx));
    CK(cuMemFree_v2(dy));
    CK(cuDevicePrimaryCtxRelease_v2(dev));
    free(hx);
    free(hy_init);
    free(hy_out);
    free(expected);
    return NULL;
}

static long run_peer_pipeline(const char* path) {
    const size_t bytes = (size_t)N * sizeof(float);
    float* hx0 = malloc(bytes);
    float* hy0 = malloc(bytes);
    float* hy1 = malloc(bytes);
    float* hy1_out = malloc(bytes);
    float* expected = malloc(bytes);
    if (!hx0 || !hy0 || !hy1 || !hy1_out || !expected) {
        fprintf(stderr, "oom in peer pipeline\n");
        exit(2);
    }
    for (size_t i = 0; i < N; i++) {
        hx0[i] = (float)(i & 127u);
        hy0[i] = (float)((i * 5u + 1u) & 63u);
        hy1[i] = (float)((i * 7u + 3u) & 63u);
    }

    CUdevice dev0, dev1;
    CUcontext ctx0, ctx1;
    CK(cuDeviceGet(&dev0, 0));
    CK(cuDeviceGet(&dev1, 1));
    CK(cuDevicePrimaryCtxRetain(&ctx0, dev0));
    CK(cuDevicePrimaryCtxRetain(&ctx1, dev1));

    int can01 = 0, can10 = 0;
    CK(cuDeviceCanAccessPeer(&can01, dev0, dev1));
    CK(cuDeviceCanAccessPeer(&can10, dev1, dev0));
    if (can01) {
        CK(cuCtxSetCurrent(ctx0));
        CUresult pr = cuCtxEnablePeerAccess(ctx1, 0);
        if (pr && pr != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
            fprintf(stderr, "FAIL enable peer 0->1: %d %s\n", pr, ename(pr));
            exit(1);
        }
    }
    if (can10) {
        CK(cuCtxSetCurrent(ctx1));
        CUresult pr = cuCtxEnablePeerAccess(ctx0, 0);
        if (pr && pr != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
            fprintf(stderr, "FAIL enable peer 1->0: %d %s\n", pr, ename(pr));
            exit(1);
        }
    }

    CUmodule mod0, mod1;
    CUfunction f0, f1;
    CUstream s0, s1;
    CUdeviceptr dx0, dy0, dx1, dy1;

    CK(cuCtxSetCurrent(ctx0));
    CK(cuModuleLoad(&mod0, path));
    CK(cuModuleGetFunction(&f0, mod0, "saxpy"));
    CK(cuStreamCreate(&s0, 0));
    CK(cuMemAlloc_v2(&dx0, bytes));
    CK(cuMemAlloc_v2(&dy0, bytes));
    CK(cuMemcpyHtoD_v2(dx0, hx0, bytes));
    CK(cuMemcpyHtoD_v2(dy0, hy0, bytes));

    CK(cuCtxSetCurrent(ctx1));
    CK(cuModuleLoad(&mod1, path));
    CK(cuModuleGetFunction(&f1, mod1, "saxpy"));
    CK(cuStreamCreate(&s1, 0));
    CK(cuMemAlloc_v2(&dx1, bytes));
    CK(cuMemAlloc_v2(&dy1, bytes));
    CK(cuMemcpyHtoD_v2(dy1, hy1, bytes));

    int n = (int)N;
    float a0 = 2.0f, a1 = 3.0f;
    void* args0[] = {&a0, &dx0, &dy0, &n};
    void* args1[] = {&a1, &dx1, &dy1, &n};

    CK(cuCtxSetCurrent(ctx0));
    CK(cuLaunchKernel(f0, GRID, 1, 1, BLOCK, 1, 1, 0, s0, args0, NULL));
    CK(cuStreamSynchronize(s0));

    CK(cuCtxSetCurrent(ctx1));
    CK(cuMemcpyPeerAsync(dx1, ctx1, dy0, ctx0, bytes, s1));
    CK(cuLaunchKernel(f1, GRID, 1, 1, BLOCK, 1, 1, 0, s1, args1, NULL));
    CK(cuStreamSynchronize(s1));
    CK(cuMemcpyDtoH_v2(hy1_out, dy1, bytes));

    for (size_t i = 0; i < N; i++) {
        float y0_after = a0 * hx0[i] + hy0[i];
        expected[i] = a1 * y0_after + hy1[i];
    }
    long bad = 0;
    check_array("peer-pipeline", 1, hy1_out, expected, N, &bad);

    CK(cuStreamDestroy_v2(s1));
    CK(cuMemFree_v2(dx1));
    CK(cuMemFree_v2(dy1));
    CK(cuModuleUnload(mod1));

    CK(cuCtxSetCurrent(ctx0));
    CK(cuStreamDestroy_v2(s0));
    CK(cuMemFree_v2(dx0));
    CK(cuMemFree_v2(dy0));
    CK(cuModuleUnload(mod0));

    CK(cuDevicePrimaryCtxRelease_v2(dev1));
    CK(cuDevicePrimaryCtxRelease_v2(dev0));
    free(hx0);
    free(hy0);
    free(hy1);
    free(hy1_out);
    free(expected);
    return bad;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: multi-gpu MODULE_FILE\n");
        return 2;
    }
    const char* path = argv[1];
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        perror(path);
        return 2;
    }
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* image = calloc((size_t)size + 1, 1);
    if (!image || fread(image, 1, (size_t)size, fp) != (size_t)size) {
        perror("read");
        return 2;
    }
    fclose(fp);

    const char* libname = getenv("DRVRUN_LIBCUDA") ? getenv("DRVRUN_LIBCUDA") : "libcuda.so.1";
    lib = dlopen(libname, RTLD_NOW | RTLD_GLOBAL);
    if (!lib) {
        fprintf(stderr, "dlopen %s: %s\n", libname, dlerror());
        return 2;
    }
#define LOAD(name) name = sym(#name)
    LOAD(cuInit);
    LOAD(cuDeviceGetCount);
    LOAD(cuDeviceGet);
    LOAD(cuDeviceGetName);
    LOAD(cuDeviceComputeCapability);
    LOAD(cuDeviceCanAccessPeer);
    LOAD(cuDevicePrimaryCtxRetain);
    LOAD(cuDevicePrimaryCtxRelease_v2);
    LOAD(cuCtxSetCurrent);
    LOAD(cuCtxEnablePeerAccess);
    LOAD(cuModuleLoad);
    LOAD(cuModuleLoadData);
    LOAD(cuModuleUnload);
    LOAD(cuModuleGetFunction);
    LOAD(cuLibraryLoadData);
    LOAD(cuLibraryGetKernel);
    LOAD(cuKernelGetFunction);
    LOAD(cuLibraryUnload);
    LOAD(cuMemAlloc_v2);
    LOAD(cuMemFree_v2);
    LOAD(cuMemcpyHtoD_v2);
    LOAD(cuMemcpyDtoH_v2);
    LOAD(cuMemcpyPeerAsync);
    LOAD(cuStreamCreate);
    LOAD(cuStreamDestroy_v2);
    LOAD(cuStreamSynchronize);
    LOAD(cuStreamBeginCapture_v2);
    LOAD(cuStreamEndCapture);
    LOAD(cuGraphInstantiateWithFlags);
    LOAD(cuGraphLaunch);
    LOAD(cuGraphExecDestroy);
    LOAD(cuGraphDestroy);
    LOAD(cuLaunchKernel);
    LOAD(cuGetErrorName);

    CK(cuInit(0));
    int count = 0;
    CK(cuDeviceGetCount(&count));
    if (count < 2) {
        printf("SKIP multi-gpu: need >= 2 GPUs (found %d)\n", count);
        free(image);
        return 77;
    }
    for (int d = 0; d < 2; d++) {
        CUdevice dev;
        char name[128];
        int major = 0, minor = 0;
        CK(cuDeviceGet(&dev, d));
        CK(cuDeviceGetName(name, sizeof name, dev));
        CK(cuDeviceComputeCapability(&major, &minor, dev));
        printf("device %d: %s (sm_%d%d)\n", d, name, major, minor);
    }

    pthread_barrier_t bar;
    if (pthread_barrier_init(&bar, NULL, 2) != 0) {
        perror("pthread_barrier_init");
        return 2;
    }
    WorkerArgs args[2] = {
        {.ordinal = 0, .image = image, .alpha = 2.0f, .y_seed = 11u, .repeats = 4, .bar = &bar},
        {.ordinal = 1, .image = image, .alpha = 3.0f, .y_seed = 29u, .repeats = 4, .bar = &bar},
    };
    pthread_t th[2];
    for (int i = 0; i < 2; i++) {
        if (pthread_create(&th[i], NULL, concurrent_worker, &args[i]) != 0) {
            perror("pthread_create");
            return 2;
        }
    }
    for (int i = 0; i < 2; i++) {
        pthread_join(th[i], NULL);
    }
    pthread_barrier_destroy(&bar);

    long peer_bad = run_peer_pipeline(path);
    long total_bad = args[0].direct_bad + args[0].graph_bad + args[0].library_bad
                   + args[1].direct_bad + args[1].graph_bad + args[1].library_bad + peer_bad;
    printf("multi-gpu: dev0=(direct=%ld,graph=%ld,library=%ld) dev1=(direct=%ld,graph=%ld,library=%ld) peer=%ld\n",
           args[0].direct_bad, args[0].graph_bad, args[0].library_bad,
           args[1].direct_bad, args[1].graph_bad, args[1].library_bad, peer_bad);
    free(image);
    return total_bad ? 1 : 0;
}
