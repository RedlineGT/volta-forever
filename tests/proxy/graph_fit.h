/* The Linux CUDA driver ABI used by the offline graph test. */
#include <stddef.h>
#include <stdint.h>
typedef struct {
    void *function;
    unsigned grid[3], block[3], shared;
    void **args, **extra;
} test_kernel_v1;
typedef struct {
    test_kernel_v1 v1;
    void *kernel, *context;
} test_kernel_v2;
typedef struct {
    int type, reserved[3];
    union { test_kernel_v2 kernel; uint64_t words[29]; } params;
    uint64_t tail;
} test_node;
_Static_assert(sizeof(test_kernel_v1)==56 && sizeof(test_kernel_v2)==72 && sizeof(test_node)==256,"64-bit driver ABI");
