// A program for driver_calls.py that reaches the driver as the CUDA runtime does, through the
// pointers that cuGetProcAddress_v2 gives for each function ("lookup"), or by the functions'
// exported names ("names"): two contexts, a module in each, launches on streams, on the legacy
// stream and on the thread's own default stream, events recorded, queried and waited for, streams
// queried and synchronised, a graph captured in one context while the other launches, launched on a
// stream and on the null stream, a module loaded and unloaded, streams and a context destroyed.
// Nothing faults. Against fault_mock_driver.c, behind driver_calls.py's recorder.
//
// usage: driver_calls PROXY lookup|names
// Prints "results N", N the bitwise OR of every call's result (0 expected).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
static void *cuda;
static int by_lookup;
static CUresult (*get_proc)(const char *, void **, int, uint64_t, int *);

static void *fn(const char *name) {
    void *p = NULL;
    if (by_lookup) {
        int status = 0;
        if (get_proc(name, &p, 12000, 0, &status) || !p) {
            fprintf(stderr, "lookup of %s failed\n", name);
            exit(2);
        }
    } else if (!(p = dlsym(cuda, name))) {
        fprintf(stderr, "no %s\n", name);
        exit(2);
    }
    return p;
}

static const char PTX[] = ".version 7.0\n.target sm_75\n.address_size 64\n"
                          ".visible .entry add_one(.param .u64 p)\n{\n ret;\n}\n"
                          ".visible .entry write_null(.param .u64 p)\n{\n ret;\n}\n";

typedef CUresult (*launch_t)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *,
                             void **, void **);

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    cuda = dlopen(argv[1], RTLD_NOW);
    if (!cuda) return 2;
    by_lookup = !strcmp(argv[2], "lookup");
    get_proc = dlsym(cuda, "cuGetProcAddress_v2");
    if (!get_proc) return 2;
    CUresult (*init)(unsigned) = fn("cuInit");
    CUresult (*ctx_create)(void **, unsigned, int) = fn("cuCtxCreate_v2");
    CUresult (*ctx_destroy)(void *) = fn("cuCtxDestroy_v2");
    CUresult (*ctx_set)(void *) = fn("cuCtxSetCurrent");
    CUresult (*ctx_sync)(void) = fn("cuCtxSynchronize");
    CUresult (*load)(void **, const void *) = fn("cuModuleLoadData");
    CUresult (*unload)(void *) = fn("cuModuleUnload");
    CUresult (*get_function)(void **, void *, const char *) = fn("cuModuleGetFunction");
    launch_t launch = fn("cuLaunchKernel");
    launch_t launch_ptsz = fn("cuLaunchKernel_ptsz");
    CUresult (*stream_create)(void **, unsigned) = fn("cuStreamCreate");
    CUresult (*stream_destroy)(void *) = fn("cuStreamDestroy_v2");
    CUresult (*stream_sync)(void *) = fn("cuStreamSynchronize");
    CUresult (*stream_sync_ptsz)(void *) = fn("cuStreamSynchronize_ptsz");
    CUresult (*stream_query)(void *) = fn("cuStreamQuery");
    CUresult (*event_create)(void **, unsigned) = fn("cuEventCreate");
    CUresult (*event_record)(void *, void *) = fn("cuEventRecord");
    CUresult (*event_sync)(void *) = fn("cuEventSynchronize");
    CUresult (*event_query)(void *) = fn("cuEventQuery");
    CUresult (*begin_capture)(void *, int) = fn("cuStreamBeginCapture_v2");
    CUresult (*end_capture)(void *, void **) = fn("cuStreamEndCapture");
    CUresult (*instantiate)(void **, void *, unsigned long long) = fn("cuGraphInstantiateWithFlags");
    CUresult (*graph_launch)(void *, void *) = fn("cuGraphLaunch");
    CUresult (*exec_destroy)(void *) = fn("cuGraphExecDestroy");
    CUresult (*graph_destroy)(void *) = fn("cuGraphDestroy");
    if (init(0)) return 3;
    int sum = 0;
    void *ctx[2], *mod[2], *add[2], *s[2], *ev[2];
    for (int i = 0; i < 2; i++) {
        sum |= ctx_create(&ctx[i], 0, 0);
        sum |= load(&mod[i], PTX);
        sum |= get_function(&add[i], mod[i], "add_one");
        sum |= stream_create(&s[i], 1);
        sum |= event_create(&ev[i], 0);
    }
    unsigned long long word = 0;
    void *args[] = {&word};
    for (int round = 0; round < 2; round++) {
        for (int i = 0; i < 2; i++) {
            sum |= ctx_set(ctx[i]);
            sum |= launch(add[i], 1, 1, 1, 32, 1, 1, 0, NULL, args, NULL);
            sum |= launch(add[i], 2, 1, 1, 32, 1, 1, 0, s[i], args, NULL);
            sum |= launch_ptsz(add[i], 3, 1, 1, 32, 1, 1, 0, NULL, args, NULL);
            sum |= event_record(ev[i], s[i]);
        }
        for (int i = 0; i < 2; i++) {
            sum |= ctx_set(ctx[i]);
            sum |= event_query(ev[i]);
            sum |= event_sync(ev[i]);
            sum |= stream_query(s[i]);
            sum |= stream_sync(s[i]);
            sum |= stream_sync(NULL);
            sum |= stream_sync_ptsz(NULL);
            sum |= ctx_sync();
        }
        // A graph in context 0, captured while context 1 launches on its own stream.
        sum |= ctx_set(ctx[0]);
        void *graph = NULL, *exec = NULL;
        sum |= begin_capture(s[0], 2);
        sum |= launch(add[0], 4, 1, 1, 32, 1, 1, 0, s[0], args, NULL);
        sum |= ctx_set(ctx[1]);
        sum |= launch(add[1], 5, 1, 1, 32, 1, 1, 0, s[1], args, NULL);
        sum |= stream_sync(s[1]);
        sum |= ctx_set(ctx[0]);
        sum |= end_capture(s[0], &graph);
        sum |= instantiate(&exec, graph, 0);
        sum |= graph_launch(exec, s[0]);
        sum |= graph_launch(exec, NULL);
        sum |= stream_sync(s[0]);
        sum |= exec_destroy(exec);
        sum |= graph_destroy(graph);
    }
    void *extra_mod;
    sum |= load(&extra_mod, PTX);
    sum |= unload(extra_mod);
    for (int i = 0; i < 2; i++) sum |= stream_destroy(s[i]);
    sum |= ctx_destroy(ctx[1]);
    sum |= ctx_set(ctx[0]);
    sum |= launch(add[0], 6, 1, 1, 32, 1, 1, 0, NULL, args, NULL);
    sum |= ctx_sync();
    printf("results %d\n", sum);
    return sum ? 1 : 0;
}
