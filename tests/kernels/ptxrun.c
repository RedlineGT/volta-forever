// ptxrun: load one PTX module through the CUDA driver API, launch one kernel of it with one block,
// and compare what each lane stored with the words given.
//
// usage: ptxrun [-d ORDINAL] [-t THREADS] [-i INT32] MODULE.ptx KERNEL WORD0 [WORD1 [WORD2 [WORD3]]]
//
// The first kernel parameter is the address of THREADS * 16 bytes (zero when it starts), of which
// each lane writes its own 16 (lane * 16). With -i the second parameter is a signed integer.
// WORDn: what every lane must have as its word n, in
// hexadecimal, or `-` for a word that is not looked at. Environment: DRVRUN_LIBCUDA (library to
// load, default libcuda.so.1).
//
// For kernels that only exist as PTX text (a register family declared again in a scope, an
// accumulator that is also an operand: tests/kernels/ptx-cases/), which no compiler of CUDA C
// writes.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
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
enum { CU_JIT_INFO_LOG_BUFFER = 3, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4, CU_JIT_ERROR_LOG_BUFFER = 5,
       CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6 };

static void* lib;
#define SYM(ret, name, ...) static ret (*name)(__VA_ARGS__)
SYM(CUresult, cuInit, unsigned);
SYM(CUresult, cuDeviceGet, CUdevice*, int);
SYM(CUresult, cuCtxCreate_v2, CUcontext*, unsigned, CUdevice);
SYM(CUresult, cuCtxDestroy_v2, CUcontext);
SYM(CUresult, cuCtxSynchronize, void);
SYM(CUresult, cuModuleLoadDataEx, CUmodule*, const void*, unsigned, int*, void**);
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

enum { WORDS = 4 };

int main(int argc, char** argv) {
    int ordinal = 0, arg = 1;
    if (arg + 1 < argc && !strcmp(argv[arg], "-d")) { ordinal = atoi(argv[arg + 1]); arg += 2; }
    int threads = 32;
    if (arg + 1 < argc && !strcmp(argv[arg], "-t")) {
        char* end;
        long n = strtol(argv[arg + 1], &end, 10);
        if (*end || n < 1 || n > 1024) { fprintf(stderr, "invalid block size\n"); return 2; }
        threads = (int)n;
        arg += 2;
    }
    int32_t integer = 0;
    if (arg + 1 < argc && !strcmp(argv[arg], "-i")) {
        char* end;
        errno = 0;
        long long n = strtoll(argv[arg + 1], &end, 10);
        if (errno || end == argv[arg + 1] || *end || n < INT32_MIN || n > INT32_MAX) {
            fprintf(stderr, "invalid signed integer\n");
            return 2;
        }
        integer = (int32_t)n;
        arg += 2;
    }
    if (argc - arg < 3 || argc - arg > 2 + WORDS) {
        fprintf(stderr, "usage: ptxrun [-d ORDINAL] [-t THREADS] [-i INT32] MODULE.ptx KERNEL WORD0 [WORD1 [WORD2 [WORD3]]]\n");
        return 2;
    }
    const char *path = argv[arg], *kernel = argv[arg + 1];
    int looked[WORDS] = {0};
    uint32_t want[WORDS] = {0};
    for (int w = 0; w < argc - arg - 2; w++) {
        const char* text = argv[arg + 2 + w];
        if (strcmp(text, "-")) { looked[w] = 1; want[w] = (uint32_t)strtoul(text, NULL, 16); }
    }

    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return 2; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* image = calloc(1, (size_t)size + 1);
    if (fread(image, 1, (size_t)size, f) != (size_t)size) { perror(path); return 2; }
    fclose(f);

    const char* libname = getenv("DRVRUN_LIBCUDA") ? getenv("DRVRUN_LIBCUDA") : "libcuda.so.1";
    lib = dlopen(libname, RTLD_NOW | RTLD_GLOBAL);
    if (!lib) { fprintf(stderr, "dlopen %s: %s\n", libname, dlerror()); return 2; }
    LOAD(cuInit); LOAD(cuDeviceGet); LOAD(cuCtxCreate_v2); LOAD(cuCtxDestroy_v2); LOAD(cuCtxSynchronize);
    LOAD(cuModuleLoadDataEx); LOAD(cuModuleGetFunction); LOAD(cuMemAlloc_v2); LOAD(cuMemcpyDtoH_v2);
    LOAD(cuMemsetD8_v2); LOAD(cuLaunchKernel); LOAD(cuGetErrorName);

    CUdevice device;
    CUcontext context;
    CK(cuInit(0));
    CK(cuDeviceGet(&device, ordinal));
    CK(cuCtxCreate_v2(&context, 0, device));

    static char info[8192], error[8192];
    int options[4] = {CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER,
                      CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void* values[4] = {info, (void*)(uintptr_t)sizeof info, error, (void*)(uintptr_t)sizeof error};
    CUmodule module;
    CUresult r = cuModuleLoadDataEx(&module, image, 4, options, values);
    if (r) { printf("FAIL the module does not load: %d %s\n%s\n", r, ename(r), error); return 1; }
    CUfunction function;
    CK(cuModuleGetFunction(&function, module, kernel));

    CUdeviceptr out;
    CK(cuMemAlloc_v2(&out, threads * WORDS * 4));
    CK(cuMemsetD8_v2(out, 0, threads * WORDS * 4));
    void* args[] = {&out, &integer};
    CK(cuLaunchKernel(function, 1, 1, 1, threads, 1, 1, 0, NULL, args, NULL));
    CK(cuCtxSynchronize());
    uint32_t got[threads][WORDS];
    CK(cuMemcpyDtoH_v2(got, out, sizeof got));

    int wrong = 0;
    for (int lane = 0; lane < threads; lane++) {
        int bad = 0;
        for (int w = 0; w < WORDS; w++) bad |= looked[w] && got[lane][w] != want[w];
        if (bad && wrong++ < 4)
            printf("  lane %2d: %08x %08x %08x %08x\n", lane, got[lane][0], got[lane][1], got[lane][2], got[lane][3]);
    }
    printf("%s %s of %s: %d of %d lanes as expected (lane 0: %08x %08x %08x %08x)\n", wrong ? "FAIL" : "PASS", kernel,
           path, threads - wrong, threads, got[0][0], got[0][1], got[0][2], got[0][3]);
    cuCtxDestroy_v2(context);
    return wrong != 0;
}
