// A stand-in for NVIDIA's driver library, for the tests of what the layer keeps over the life of a
// process (lifecycle.sh). The layer loads it as the driver (VF_REAL_LIBCUDA). Every module it
// loads is given a handle of its own, which no other load is given: what the layer keeps of a
// module is dropped only when the layer is told that the module is gone.
#include <stdint.h>

static uintptr_t new_handle(void) {
    static uintptr_t next = 16;
    return __atomic_fetch_add(&next, 1, __ATOMIC_RELAXED);
}

int cuInit(unsigned flags) { (void)flags; return 0; }
int cuModuleLoadData(void **module, const void *image) {
    (void)image;
    *module = (void *)new_handle();
    return 0;
}
int cuModuleUnload(void *module) { (void)module; return 0; }
