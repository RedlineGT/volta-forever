/* Fresh processes isolate cold capture from the case where an ordinary launch
 * already caused a second translation. Graph getters must return the original
 * function, and an exec-node update must accept the node from the original graph. */
#include "graph_fit.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *proxy;

static void *symbol(const char *name) {
    void *entry = dlsym(proxy, name);
    if (!entry) {
        fprintf(stderr, "missing %s\n", name);
        exit(2);
    }
    return entry;
}

#define CALL(name, type, ...) ((int (*) type)symbol(#name))(__VA_ARGS__)
#define CHECK(call) do { \
    int err = (call); \
    if (err) { fprintf(stderr, "%s returned %d\n", #call, err); return 1; } \
} while (0)

static void *read_fixture(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) return NULL;
    long size = ftell(file);
    if (size <= 0 || fseek(file, 0, SEEK_SET)) return NULL;
    void *bytes = malloc((size_t)size);
    if (!bytes || fread(bytes, 1, (size_t)size, file) != (size_t)size) return NULL;
    fclose(file);
    return bytes;
}

static int launch_large(void *function, void *stream, void **args) {
    typedef int (*launch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                          unsigned, void *, void **, void **);
    return ((launch)symbol("cuLaunchKernel"))(function, 1, 1, 1, 1024, 1, 1, 0, stream, args, NULL);
}

int main(int argc, char **argv) {
    if (argc != 5) return 2;
    proxy = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!proxy) {
        fprintf(stderr, "%s\n", dlerror());
        return 2;
    }
    void *bytes = read_fixture(argv[2]);
    if (!bytes) return 2;

    void *mod = NULL, *function = NULL, *graph = NULL, *node = NULL, *exec = NULL, *stream = (void *)42;
    CHECK(CALL(cuModuleLoadData, (void **, const void *), &mod, bytes));
    CHECK(CALL(cuModuleGetFunction, (void **, void *, const char *), &function, mod, "k_fit_regs"));
    int regs = 0;
    CHECK(CALL(cuFuncGetAttribute, (int *, int, void *), &regs, 4, function));
    const char *mode = argv[3];
    int expected = atoi(argv[4]);
    printf("%s regs=%d\n", mode, regs);

    int set = !strcmp(mode, "set"), update = !strcmp(mode, "update"), small = !strcmp(mode, "small");
    int n = 17;
    void *args[] = { &n };
    test_kernel_v2 p = { .v1 = { function, {1, 1, 1}, {small || set || update ? 256 : 1024, 1, 1}, 0, args, NULL } };
    if (!strcmp(mode, "capture") || !strcmp(mode, "warmcapture")) {
        if (!strcmp(mode, "warmcapture")) CHECK(launch_large(function, stream, args));
        CHECK(CALL(cuStreamBeginCapture_v2, (void *, int), stream, 1));
        CHECK(launch_large(function, stream, args));
        CHECK(CALL(cuStreamEndCapture, (void *, void **), stream, &graph));
        size_t count = 1;
        CHECK(CALL(cuGraphGetNodes, (void *, void **, size_t *), graph, &node, &count));
        if (count != 1) return 1;
    } else {
        CHECK(CALL(cuGraphCreate, (void **, unsigned), &graph, 0));
        if (!strcmp(mode, "v1")) {
            CHECK(CALL(cuGraphAddKernelNode, (void **, void *, void **, size_t, const test_kernel_v1 *),
                &node, graph, NULL, 0, &p.v1));
        } else if (!strcmp(mode, "generic")) {
            test_node generic = {0};
            generic.params.kernel = p;
            CHECK(CALL(cuGraphAddNode, (void **, void *, void **, size_t, test_node *),
                &node, graph, NULL, 0, &generic));
            if (generic.params.kernel.v1.function != function) return 1;
        } else {
            CHECK(CALL(cuGraphAddKernelNode_v2, (void **, void *, void **, size_t, const test_kernel_v2 *),
                &node, graph, NULL, 0, &p));
        }
        if (set) {
            p.v1.block[0] = 1024;
            CHECK(CALL(cuGraphKernelNodeSetParams_v2, (void *, const test_kernel_v2 *), node, &p));
        }
    }
    test_kernel_v2 got = {0};
    CHECK(CALL(cuGraphKernelNodeGetParams_v2, (void *, test_kernel_v2 *), node, &got));
    if (got.v1.function != function || memcmp(got.v1.block, p.v1.block, sizeof(p.v1.block))) {
        fprintf(stderr, "graph getter changed function identity or block dimensions\n");
        return 1;
    }
    int result = CALL(cuGraphInstantiateWithFlags, (void **, void *, unsigned long long), &exec, graph, 0);
    if (update) {
        if (result) {
            fprintf(stderr, "small graph could not be instantiated: %d\n", result);
            return 1;
        }
        p.v1.block[0] = 1024;
        result = CALL(cuGraphExecKernelNodeSetParams_v2,
                      (void *, void *, const test_kernel_v2 *), exec, node, &p);
    }
    if (result != expected) {
        fprintf(stderr, "%s returned %d, expected %d\n", update ? "exec-set" : "instantiate", result, expected);
        return 1;
    }
    if (result == 0) CHECK(CALL(cuGraphLaunch, (void *, void *), exec, stream));
    if (exec) CHECK(CALL(cuGraphExecDestroy, (void *), exec));
    CHECK(CALL(cuGraphDestroy, (void *), graph));
    CHECK(CALL(cuModuleUnload, (void *), mod));
    free(bytes);
    return 0;
}
