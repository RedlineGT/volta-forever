// Driver for link_refusals.py: a linker whose output is the image the test puts in `linked`, with
// one handle for every link state (as an allocator may give a freed one again), and modules that
// have every kernel they are asked for.
#include <stddef.h>
#include <stdint.h>

const void *linked;
size_t linked_size;
unsigned inputs, links, destroyed;

int cuLinkCreate_v2(unsigned count, int *options, void **values, void **state) {
    ++links;
    *state = (void *)(uintptr_t)0x1000;
    return 0;
}
int cuLinkAddData_v2(void *state, int type, void *data, size_t size, const char *name, unsigned count,
                     int *options, void **values) {
    ++inputs;
    return 0;
}
int cuLinkComplete(void *state, void **image, size_t *size) {
    *image = (void *)linked;
    *size = linked_size;
    return 0;
}
int cuLinkDestroy(void *state) {
    ++destroyed;
    return 0;
}

static uintptr_t modules;
int cuModuleLoadData(void **module, const void *image) {
    *module = (void *)(0x100000 + 0x100 * ++modules);
    return 0;
}
int cuModuleUnload(void *module) { return 0; }
int cuModuleGetFunction(void **function, void *module, const char *name) {
    *function = (void *)((uintptr_t)module + 1);
    return 0;
}
