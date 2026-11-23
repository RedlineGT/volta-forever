// ldsmrun: run the kernels of k_stored.cu through the CUDA driver API and check every word they
// store against a host model of ldmatrix (the fragment layouts of the PTX ISA manual).
//
// usage: ldsmrun [-d ORDINAL] [-g BLOCKS] [-k ROUNDS] [-r RUNS] MODULE_FILE
//
// Every kernel runs RUNS times (default 3) on BLOCKS blocks of 4 warps (default 640) for ROUNDS
// rounds (default 64). Many blocks storing at once keep the memory pipeline full, so that a store
// reads its register late. Environment: DRVRUN_LIBCUDA (library to load, default libcuda.so.1).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef unsigned long long CUdeviceptr;

static void* lib;
#define SYM(ret, name, ...) static ret (*name)(__VA_ARGS__)
SYM(CUresult, cuInit, unsigned);
SYM(CUresult, cuDeviceGet, CUdevice*, int);
SYM(CUresult, cuCtxCreate_v2, CUcontext*, unsigned, CUdevice);
SYM(CUresult, cuCtxSynchronize, void);
SYM(CUresult, cuModuleLoadData, CUmodule*, const void*);
SYM(CUresult, cuModuleGetFunction, CUfunction*, CUmodule, const char*);
SYM(CUresult, cuMemAlloc_v2, CUdeviceptr*, size_t);
SYM(CUresult, cuMemcpyDtoH_v2, void*, CUdeviceptr, size_t);
SYM(CUresult, cuMemsetD8_v2, CUdeviceptr, unsigned char, size_t);
SYM(CUresult, cuLaunchKernel, CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
    void*, void**, void**);
SYM(CUresult, cuGetErrorName, CUresult, const char**);

static void* sym(const char* name) {
    void* p = dlsym(lib, name);
    if (!p) { fprintf(stderr, "missing symbol %s\n", name); exit(2); }
    return p;
}
#define LOAD(name) name = sym(#name)
static const char* ename(CUresult r) {
    const char* s = "?";
    cuGetErrorName(r, &s);
    return s ? s : "?";
}
#define CK(call) do { CUresult r_ = (call); if (r_) { printf("FAIL %s -> %d %s\n", #call, r_, ename(r_)); exit(1); } } while (0)

#define WARPS 4
// As in k_stored.cu.
static uint32_t word_of(uint32_t block, uint32_t warp, uint32_t i) {
    uint32_t x = (block * WARPS + warp) * 128u + i;
    x ^= x >> 7;
    x *= 0x9E3779B1u;
    return x ^ (x >> 15);
}
// The half of column c in row r (0 to 7) of matrix m, in round k: row r comes from lane 8m + r,
// which gives row (r + k) mod 8 of the matrix.
static uint32_t half_at(uint32_t block, uint32_t warp, int m, int r, int c, int k) {
    uint32_t w = word_of(block, warp, (m * 8 + ((r + k) & 7)) * 4 + (c >> 1));
    return (c & 1) ? w >> 16 : w & 0xffff;
}
// What thread `lane` has in register m after the load of round k.
static uint32_t expected(int trans, uint32_t block, uint32_t warp, int lane, int m, int k) {
    int group = lane >> 2, thread = lane & 3;
    if (!trans)  // the pair of halves 2 * thread, 2 * thread + 1 of row `group`
        return half_at(block, warp, m, group, 2 * thread, k) | half_at(block, warp, m, group, 2 * thread + 1, k) << 16;
    // the halves of column `group` in rows 2 * thread and 2 * thread + 1
    return half_at(block, warp, m, 2 * thread, group, k) | half_at(block, warp, m, 2 * thread + 1, group, k) << 16;
}

int main(int argc, char** argv) {
    int ordinal = 0, blocks = 640, rounds = 64, runs = 3;
    const char* path = NULL;
    for (int i = 1; i < argc; i++) {
        if (i + 1 < argc && !strcmp(argv[i], "-d")) ordinal = atoi(argv[++i]);
        else if (i + 1 < argc && !strcmp(argv[i], "-g")) blocks = atoi(argv[++i]);
        else if (i + 1 < argc && !strcmp(argv[i], "-k")) rounds = atoi(argv[++i]);
        else if (i + 1 < argc && !strcmp(argv[i], "-r")) runs = atoi(argv[++i]);
        else path = argv[i];
    }
    if (!path || blocks < 1 || rounds < 1 || runs < 1) {
        fprintf(stderr, "usage: ldsmrun [-d ORDINAL] [-g BLOCKS] [-k ROUNDS] [-r RUNS] MODULE_FILE\n");
        return 2;
    }
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return 2; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* image = calloc(1, size + 1);
    if (fread(image, 1, size, f) != (size_t)size) { perror(path); return 2; }
    fclose(f);
    const char* name = getenv("DRVRUN_LIBCUDA");
    lib = dlopen(name ? name : "libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    LOAD(cuInit); LOAD(cuDeviceGet); LOAD(cuCtxCreate_v2); LOAD(cuCtxSynchronize); LOAD(cuModuleLoadData);
    LOAD(cuModuleGetFunction); LOAD(cuMemAlloc_v2); LOAD(cuMemcpyDtoH_v2); LOAD(cuMemsetD8_v2);
    LOAD(cuLaunchKernel); LOAD(cuGetErrorName);
    CUdevice dev;
    CUcontext ctx;
    CK(cuInit(0));
    CK(cuDeviceGet(&dev, ordinal));
    CK(cuCtxCreate_v2(&ctx, 0, dev));
    CUmodule mod;
    CK(cuModuleLoadData(&mod, image));
    size_t threads = (size_t)blocks * WARPS * 32, words = threads * rounds * 4;
    CUdeviceptr out;
    CK(cuMemAlloc_v2(&out, words * 4));
    uint32_t* host = malloc(words * 4);
    const char* kernels[] = { "t_stored_plain", "t_stored_trans" };
    int bad_kernels = 0;
    for (int n = 0; n < 2; n++) {
        CUfunction fn;
        CK(cuModuleGetFunction(&fn, mod, kernels[n]));
        long wrong = 0, shown = 0;
        for (int run = 0; run < runs; run++) {
            CK(cuMemsetD8_v2(out, 0xff, words * 4));
            void* params[] = { &out, &rounds };
            CK(cuLaunchKernel(fn, blocks, 1, 1, WARPS * 32, 1, 1, 0, NULL, params, NULL));
            CK(cuCtxSynchronize());
            CK(cuMemcpyDtoH_v2(host, out, words * 4));
            for (int k = 0; k < rounds; k++)
                for (size_t t = 0; t < threads; t++)
                    for (int m = 0; m < 4; m++) {
                        uint32_t want = expected(n, t / (WARPS * 32), (t / 32) % WARPS, t % 32, m, k);
                        uint32_t got = host[((size_t)k * threads + t) * 4 + m];
                        if (got != want) {
                            if (shown++ < 4)
                                printf("  %s run %d round %d thread %zu register %d: %08x, expected %08x\n", kernels[n], run, k, t, m, got, want);
                            wrong++;
                        }
                    }
        }
        printf("%s: %zu words in %d runs, %ld wrong\n", kernels[n], words * runs, runs, wrong);
        bad_kernels += wrong != 0;
    }
    printf("2 kernels, %d wrong\n", bad_kernels);
    return bad_kernels != 0;
}
