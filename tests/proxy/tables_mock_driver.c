// Driver for readonly_tables.py. Keeps the text of every image it is given, by handle: a kernel or a
// variable is found in a library when its name is in the library's text. The module that the driver
// gives for a library (cuLibraryGetModule, or the module of a function of one of its kernels) is the
// library's handle plus MODULE; a kernel is the library's handle times 16 plus 1, a function the
// kernel plus FUNCTION.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { MAX = 4096, MODULE = 100000, FUNCTION = 1 << 30 };
static char *text[MAX];
unsigned loads;
uintptr_t last;

int cuLibraryLoadData(void **lib, const void *image, int *jo, void **jv, unsigned jc,
                      int *opts, void **values, unsigned count) {
    (void)jo, (void)jv, (void)jc, (void)opts, (void)values, (void)count;
    if (++loads >= MAX) return 2;
    text[loads] = strdup(image);
    last = loads;
    *lib = (void *)(uintptr_t)loads;
    return 0;
}
int cuLibraryUnload(void *lib) {
    uintptr_t h = (uintptr_t)lib;
    if (h < MAX) { free(text[h]); text[h] = 0; }
    return 0;
}
/* The text of the image last loaded, and of the image loaded as `lib`. */
const char *last_image(void) { return text[last] ? text[last] : ""; }
const char *image_of(uintptr_t lib) { return lib < MAX && text[lib] ? text[lib] : ""; }

static int has(uintptr_t lib, const char *what, const char *name) {
    if (lib >= MAX || !text[lib]) return 0;
    char needle[512];
    strcpy(needle, what);
    strncat(needle, name, sizeof needle - strlen(needle) - 2);
    return strstr(text[lib], needle) != 0;
}
int cuLibraryGetKernel(void **kernel, void *lib, const char *name) {
    if (!has((uintptr_t)lib, ".entry ", name)) { *kernel = 0; return 500; }
    *kernel = (void *)((uintptr_t)lib * 16 + 1);
    return 0;
}
int cuKernelGetName(const char **name, void *kernel) { (void)kernel; *name = "k"; return 0; }
int cuKernelGetFunction(void **func, void *kernel) { *func = (void *)((uintptr_t)kernel + FUNCTION); return 0; }
int cuKernelGetLibrary(void **lib, void *kernel) { *lib = (void *)((uintptr_t)kernel / 16); return 0; }
int cuFuncGetModule(void **mod, void *func) {
    *mod = (void *)(((uintptr_t)func - FUNCTION) / 16 + MODULE);
    return 0;
}
int cuLibraryGetModule(void **mod, void *lib) { *mod = (void *)((uintptr_t)lib + MODULE); return 0; }
static int global(unsigned long long *dptr, size_t *size, uintptr_t lib, const char *name) {
    if (!has(lib, " ", name)) return 500;
    if (dptr) *dptr = 0x100000ull * lib;
    if (size) *size = 8;
    return 0;
}
int cuLibraryGetGlobal(unsigned long long *dptr, size_t *size, void *lib, const char *name) {
    return global(dptr, size, (uintptr_t)lib, name);
}
int cuModuleGetGlobal(unsigned long long *dptr, size_t *size, void *mod, const char *name) {
    return global(dptr, size, (uintptr_t)mod - MODULE, name);
}
int cuModuleGetGlobal_v2(unsigned long long *dptr, size_t *size, void *mod, const char *name) {
    return global(dptr, size, (uintptr_t)mod - MODULE, name);
}
