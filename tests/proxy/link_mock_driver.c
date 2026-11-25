// Driver for link_refusals.py and nccl_remedy.py: a linker whose output is the image the test puts
// in `linked`, with one handle for every link state (as an allocator may give a freed one again),
// and modules that have every kernel they are asked for. What another thread may do while the
// driver destroys a link or unloads a module, once the handle is free (`while_freed`), is done
// there; the next module loaded can be given a handle chosen by the test (`next_module`). An unload
// fails with 201 while `unload_fails` is set, a link destroy with 400 while `destroy_fails` is.
// With MOCK_LOAD_SLEEP=SECONDS in the environment, a module load says when it begins and ends on
// standard error and takes that long between, as the driver's compilation of a large image does
// (reported_traps.py: what the layer says comes before it).
#define _GNU_SOURCE
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

const void *linked;
size_t linked_size;
unsigned inputs, links, destroyed;
void (*while_freed)(void);
uintptr_t next_module;
unsigned unload_fails, destroy_fails;

int cuLinkCreate_v2(unsigned count, int *options, void **values, void **state) {
    ++links;
    *state = (void *)(uintptr_t)0x1000;
    return 0;
}
/* The .target line of the last PTX input. */
char seen_target[32];
int cuLinkAddData_v2(void *state, int type, void *data, size_t size, const char *name, unsigned count,
                     int *options, void **values) {
    ++inputs;
    const char *line = type == 1 ? memmem(data, size, ".target", 7) : 0;
    memset(seen_target, 0, sizeof seen_target);
    for (size_t n = 0; line && n < sizeof seen_target - 1 && line[n] != '\n'; ++n) seen_target[n] = line[n];
    return 0;
}
int cuLinkComplete(void *state, void **image, size_t *size) {
    *image = (void *)linked;
    if (size) *size = linked_size;
    return 0;
}
int cuLinkDestroy(void *state) {
    if (destroy_fails) return 400;
    ++destroyed;
    if (while_freed) while_freed();
    return 0;
}

static uintptr_t modules;
int cuModuleLoadData(void **module, const void *image) {
    const char *slow = getenv("MOCK_LOAD_SLEEP");
    if (slow) {
        fprintf(stderr, "mock: cuModuleLoadData begins\n");
        usleep((useconds_t)(atof(slow) * 1e6));
        fprintf(stderr, "mock: cuModuleLoadData returns\n");
    }
    *module = (void *)(next_module ? next_module : 0x100000 + 0x100 * ++modules);
    next_module = 0;
    return 0;
}
int cuModuleUnload(void *module) {
    if (unload_fails) return 201;
    if (while_freed) while_freed();
    return 0;
}
int cuModuleGetFunction(void **function, void *module, const char *name) {
    *function = (void *)((uintptr_t)module + 1);
    return 0;
}
int cuModuleLoadDataEx(void **module, const void *image, unsigned count, int *options, void **values) {
    return cuModuleLoadData(module, image);
}
int cuModuleLoadFatBinary(void **module, const void *image) { return cuModuleLoadData(module, image); }
int cuModuleLoad(void **module, const char *path) { return cuModuleLoadData(module, path); }
/* Libraries, with every kernel they are asked for. */
int cuLibraryLoadData(void **library, const void *image, int *jo, void **jv, unsigned jc, int *lo, void **lv,
                      unsigned lc) {
    return cuModuleLoadData(library, image);
}
int cuLibraryLoadFromFile(void **library, const char *path, int *jo, void **jv, unsigned jc, int *lo, void **lv,
                          unsigned lc) {
    return cuModuleLoadData(library, path);
}
int cuLibraryUnload(void *library) { return cuModuleUnload(library); }
int cuLibraryGetKernel(void **kernel, void *library, const char *name) {
    *kernel = (void *)((uintptr_t)library + 2);
    return 0;
}
