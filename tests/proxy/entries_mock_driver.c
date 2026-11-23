// Driver for library_entries.py. Entry 1 is refused by the layer and entry 2 works. Its compiler
// refuses PTX with a line that holds 12345 (error 218, and the line in the error log, as ptxas
// writes it); a lookup of a name with "absent" in it finds nothing (500, and NULL).
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

unsigned loads, unloads, preserved;
uintptr_t next_handle;
unsigned force_name_fallback, invalid_function_names;
enum { FUNCTION_TAG = 0x10000 };
static unsigned only_good[256];
static void *loaded(void) {
    ++loads;
    uintptr_t handle = next_handle ? next_handle : loads;
    next_handle = 0;
    return (void *)handle;
}

/* The compiler takes this long (milliseconds) to refuse. */
unsigned refusal_delay;
static int compile(const char *text, int *jo, void **jv, unsigned jc) {
    const char *hit = strstr(text, "12345");
    if (!hit) return 0;
    if (refusal_delay) usleep(refusal_delay * 1000);
    int line = 1;
    for (const char *p = text; p < hit; ++p) line += *p == '\n';
    char *buffer = 0;
    size_t size = 0;
    for (unsigned i = 0; i < jc; ++i) {
        if (jo[i] == 5) buffer = jv[i];
        if (jo[i] == 6) size = (size_t)(uintptr_t)jv[i];
    }
    if (buffer && size)
        snprintf(buffer, size, "ptxas application ptx input, line %d; error   : Feature 'x' requires .target sm_80\n",
                 line);
    return 218;
}

int cuLibraryLoadData(void **lib, const void *image, int *jo, void **jv, unsigned jc,
                      int *opts, void **values, unsigned count) {
    if (compile(image, jo, jv, jc)) return 218;
    for (unsigned i = 0; i < count; ++i) preserved += opts[i] == 1;
    *lib = loaded();
    // The proxy passes ordinary PTX in this fixture. A lazy submodule contains
    // only the good entry, whereas the materialized library contains both.
    only_good[(uintptr_t)*lib] = strstr(image, "good(") && !strstr(image, "bad(");
    return 0;
}
int cuModuleLoadData(void **mod, const void *image) {
    *mod = loaded();
    only_good[(uintptr_t)*mod] = 0;
    return 0;
}
int cuLibraryUnload(void *lib) { ++unloads; return 0; }
int cuModuleUnload(void *mod) { ++unloads; return 0; }
int cuLibraryGetModule(void **mod, void *lib) {
    *mod = (void *)((uintptr_t)lib + 32);
    only_good[(uintptr_t)*mod] = only_good[(uintptr_t)lib];
    return 0;
}
int cuLibraryGetKernelCount(unsigned *count, void *lib) { *count = 2 - only_good[(uintptr_t)lib]; return 0; }
int cuModuleGetFunctionCount(unsigned *count, void *mod) { *count = 2 - only_good[(uintptr_t)mod]; return 0; }

static void *entry(void *owner, unsigned index) {
    return (void *)((uintptr_t)owner * 16 + index);
}
static int list(void **out, unsigned cap, void *owner) {
    if (only_good[(uintptr_t)owner]) {
        if (cap) out[0] = entry(owner, 2);
    } else {
        for (unsigned i = 0; i < cap && i < 2; ++i) out[i] = entry(owner, i + 1);
    }
    return 0;
}
int cuLibraryEnumerateKernels(void **out, unsigned cap, void *lib) { return list(out, cap, lib); }
int cuModuleEnumerateFunctions(void **out, unsigned cap, void *mod) {
    int rc = list(out, cap, mod);
    for (unsigned i = 0; i < cap && i < 2 - only_good[(uintptr_t)mod]; ++i)
        out[i] = (void *)((uintptr_t)out[i] | FUNCTION_TAG);
    return rc;
}
int cuLibraryGetKernel(void **kernel, void *lib, const char *name) {
    if (strstr(name, "absent")) {
        *kernel = 0;
        return 500;
    }
    *kernel = entry(lib, strcmp(name, "good") == 0 ? 2 : 1);
    return 0;
}
int cuKernelGetName(const char **name, void *kernel) {
    if (force_name_fallback) return 801;
    *name = ((uintptr_t)kernel % 16 == 2) ? "good" : "bad";
    return 0;
}
int cuFuncGetName(const char **name, void *function) {
    if (!((uintptr_t)function & FUNCTION_TAG)) { ++invalid_function_names; return 400; }
    *name = ((uintptr_t)function % 16 == 2) ? "good" : "bad";
    return 0;
}
int cuGetProcAddress(const char *symbol, void **ptr, int version, uint64_t flags) {
    if (strcmp(symbol, "cuModuleEnumerateFunctions") == 0) *ptr = cuModuleEnumerateFunctions;
    else if (strcmp(symbol, "cuLibraryEnumerateKernels") == 0) *ptr = cuLibraryEnumerateKernels;
    else return 500;
    return 0;
}
int cuKernelGetFunction(void **func, void *kernel) { *func = (void *)((uintptr_t)kernel | FUNCTION_TAG); return 0; }
int cuKernelGetLibrary(void **lib, void *kernel) { *lib = (void *)((uintptr_t)kernel / 16); return 0; }
int cuFuncGetModule(void **mod, void *func) { return cuLibraryGetModule(mod, (void *)(((uintptr_t)func & ~((uintptr_t)FUNCTION_TAG)) / 16)); }
