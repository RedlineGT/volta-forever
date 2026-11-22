// Peer-copy check for two CUDA devices visible to one process.
// Build with gcc -O2 -I"$CUDA_HOME/include" peer-copy.c -L"$CUDA_HOME/lib64/stubs" -lcuda -o peer-copy
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CU(call) do {                                                        \
    CUresult rc = (call);                                                    \
    if (rc != CUDA_SUCCESS) {                                                \
        const char *error = "unknown";                                       \
        cuGetErrorName(rc, &error);                                          \
        fprintf(stderr, "%s: %s (%d)\n", #call, error, rc);                  \
        exit(1);                                                             \
    }                                                                        \
} while (0)

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(void) {
    enum { devices = 2, measured = 32 };
    const size_t bytes = (size_t)64 << 20;
    CUdevice device[devices];
    CUcontext context[devices];
    CUdeviceptr src[devices], dst[devices];
    CUstream stream[devices];
    char bus[devices][32];
    unsigned char *expected = malloc(bytes), *actual = malloc(bytes);
    if (!expected || !actual) return 1;
    CU(cuInit(0));
    int count;
    CU(cuDeviceGetCount(&count));
    if (count != devices) {
        fprintf(stderr, "Select exactly two devices with CUDA_VISIBLE_DEVICES\n");
        return 2;
    }
    for (int i = 0; i < devices; ++i) {
        CU(cuDeviceGet(&device[i], i));
        CU(cuDeviceGetPCIBusId(bus[i], sizeof(bus[i]), device[i]));
        CU(cuDevicePrimaryCtxRetain(&context[i], device[i]));
        CU(cuCtxSetCurrent(context[i]));
        CU(cuMemAlloc(&src[i], bytes));
        CU(cuMemAlloc(&dst[i], bytes));
        CU(cuStreamCreate(&stream[i], CU_STREAM_NON_BLOCKING));
    }
    for (int i = 0; i < devices; ++i) {
        const int j = 1 - i;
        int access;
        CU(cuDeviceCanAccessPeer(&access, device[i], device[j]));
        if (!access) { fprintf(stderr, "No peer access from %s to %s\n", bus[i], bus[j]); return 3; }
        CU(cuCtxSetCurrent(context[i]));
        CU(cuCtxEnablePeerAccess(context[j], 0));
    }
    for (int i = 0; i < devices; ++i) {
        const int j = 1 - i;
        uint32_t word = 17 + i;
        for (size_t k = 0; k < bytes; ++k) {
            word = 1664525u * word + 1013904223u;
            expected[k] = word >> 24;
        }
        CU(cuCtxSetCurrent(context[j]));
        CU(cuMemsetD8(dst[j], 0, bytes));
        CU(cuCtxSynchronize());
        CU(cuCtxSetCurrent(context[i]));
        CU(cuMemcpyHtoD(src[i], expected, bytes));
        for (int run = 0; run < 4; ++run) {
            int n = run ? measured : 4;
            const double start = seconds();
            for (int k = 0; k < n; ++k)
                CU(cuMemcpyPeerAsync(dst[j], context[j], src[i], context[i], bytes, stream[i]));
            CU(cuStreamSynchronize(stream[i]));
            const double elapsed = seconds() - start;
            if (run) printf("%s -> %s run=%d bytes=%zu copies=%d seconds=%.9f GB/s=%.3f\n",
                            bus[i], bus[j], run, bytes, n, elapsed, bytes * (double)n / elapsed / 1e9);
        }
        CU(cuCtxSetCurrent(context[j]));
        CU(cuMemcpyDtoH(actual, dst[j], bytes));
        if (memcmp(expected, actual, bytes)) {
            fprintf(stderr, "Data mismatch for %s -> %s\n", bus[i], bus[j]); return 4;
        }
        printf("verified %zu bytes %s -> %s\n", bytes, bus[i], bus[j]);
    }
    for (int i = 0; i < devices; ++i) {
        CU(cuCtxSetCurrent(context[i]));
        CU(cuCtxDisablePeerAccess(context[1-i]));
    }
    for (int i = 0; i < devices; ++i) {
        CU(cuCtxSetCurrent(context[i]));
        CU(cuStreamDestroy(stream[i]));
        CU(cuMemFree(src[i]));
        CU(cuMemFree(dst[i]));
        CU(cuDevicePrimaryCtxRelease(device[i]));
    }
    free(expected);
    free(actual);
    return 0;
}
