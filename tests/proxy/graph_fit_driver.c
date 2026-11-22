/* Single-node stand-in for the CUDA driver. Recording a launch succeeds even
 * when its block has too many registers. Only instantiation or an executable
 * update validates the claim, just as on GV100. Read the register count from
 * the actual translated cubin the proxy submits for this test fixture. */
#include "graph_fit.h"
#include <elf.h>
#include <stdlib.h>
#include <string.h>

typedef struct { unsigned regs; } module;
typedef struct { test_kernel_v2 params; } graph;
typedef struct {
    graph *source;
    test_kernel_v2 params;
} executable;

static graph *capture;

int cuModuleLoadData(void **out, const void *bytes) {
    const Elf64_Ehdr *elf = bytes;
    if (memcmp(elf->e_ident, ELFMAG, SELFMAG) || elf->e_machine != EM_CUDA) return 200;
    const Elf64_Shdr *sections = (const void *)((const char *)bytes + elf->e_shoff);
    const char *names = (const char *)bytes + sections[elf->e_shstrndx].sh_offset;
    for (unsigned i = 1; i < elf->e_shnum; i++) {
        if (strcmp(names + sections[i].sh_name, ".text.k_fit_regs")) continue;
        module *m = malloc(sizeof(*m));
        if (!m) return 2;
        m->regs = sections[i].sh_info >> 24;
        *out = m;
        return 0;
    }
    return 500;
}

int cuModuleGetFunction(void **out, void *m, const char *name) {
    if (!m || strcmp(name, "k_fit_regs")) return 500;
    *out = m;
    return 0;
}

int cuModuleUnload(void *m) {
    free(m);
    return 0;
}

int cuFuncGetAttribute(int *out, int attr, const module *m) {
    if (attr != 4 || !out || !m) return 1;
    *out = m->regs;
    return 0;
}

int cuCtxGetDevice(int *out) {
    *out = 0;
    return 0;
}

static int validate(const test_kernel_v2 *p) {
    const module *m = p->v1.function;
    const unsigned *block = p->v1.block;
    if (!m || !block[0] || !block[1] || !block[2]) return 1;
    if (block[0] > 1024 || block[1] > 1024 || block[2] > 64) return 1;
    const unsigned threads = block[0] * block[1] * block[2];
    if (threads > 1024 || p->v1.shared > 49152) return 1;
    const unsigned warps = (((threads + 31) / 32 + 3) / 4) * 4;
    const unsigned per_warp = ((m->regs * 32 + 255) / 256) * 256;
    return warps * per_warp > 65536 ? 701 : 0;
}

int cuGraphCreate(graph **out, unsigned flags) {
    if (flags) return 1;
    *out = calloc(1, sizeof(graph));
    return *out ? 0 : 2;
}

int cuGraphAddKernelNode_v2(void **node, graph *g, void **deps, size_t n, const test_kernel_v2 *p) {
    if (!g || n || deps || !p) return 1;
    g->params = *p;
    *node = g;
    return 0;
}

int cuGraphAddKernelNode(void **node, graph *g, void **deps, size_t n, const test_kernel_v1 *p) {
    test_kernel_v2 v2 = { .v1 = *p };
    return cuGraphAddKernelNode_v2(node, g, deps, n, &v2);
}

int cuGraphAddNode(void **node, graph *g, void **deps, size_t n, test_node *p) {
    if (p->type != 0) return 1;
    return cuGraphAddKernelNode_v2(node, g, deps, n, &p->params.kernel);
}

int cuGraphKernelNodeSetParams_v2(graph *node, const test_kernel_v2 *p) {
    node->params = *p;
    return 0;
}

int cuGraphKernelNodeGetParams_v2(const graph *node, test_kernel_v2 *p) {
    *p = node->params;
    return 0;
}

int cuGraphGetNodes(graph *g, void **nodes, size_t *n) {
    if (nodes && *n) nodes[0] = g;
    *n = 1;
    return 0;
}

int cuGraphInstantiateWithFlags(executable **out, graph *g, unsigned long long flags) {
    if (flags) return 1;
    int result = validate(&g->params);
    if (result) return result;
    executable *e = malloc(sizeof(*e));
    if (!e) return 2;
    e->source = g;
    e->params = g->params;
    *out = e;
    return 0;
}

int cuGraphExecKernelNodeSetParams_v2(executable *e, graph *node, const test_kernel_v2 *p) {
    if (e->source != node) return 1;
    int result = validate(p);
    if (!result) e->params = *p;
    return result;
}

int cuGraphLaunch(const executable *e, void *stream) {
    (void)stream;
    return validate(&e->params);
}

int cuGraphExecDestroy(executable *e) { free(e); return 0; }
int cuGraphDestroy(graph *g) { free(g); return 0; }

int cuStreamBeginCapture_v2(void *stream, int mode) {
    (void)stream;
    (void)mode;
    return cuGraphCreate(&capture, 0);
}

int cuStreamEndCapture(void *stream, graph **g) {
    (void)stream;
    *g = capture;
    capture = NULL;
    return 0;
}

int cuLaunchKernel(void *f, unsigned gx, unsigned gy, unsigned gz,
                   unsigned bx, unsigned by, unsigned bz, unsigned shared,
                   void *stream, void **args, void **extra) {
    (void)stream;
    test_kernel_v2 p = { .v1 = { f, {gx, gy, gz}, {bx, by, bz}, shared, args, extra } };
    if (capture) {
        capture->params = p;
        return 0;
    }
    return validate(&p);
}
