/* Shared by the GPU tests of the bf16 cuBLAS variants (tests/gpu/forward-bf16.sh): built against the
 * CUDA 13.5 headers, run with forward/bf16tc/ first in the library path. cuBLAS 12.9's own result
 * comes from the same arguments given to libcublas.so.12 (and libcublasLt.so.12) directly, with a
 * handle made there: a handle the variants never saw, so its calls are 12.9's alone. */
#ifndef VF_BF16_COMMON_H
#define VF_BF16_COMMON_H
#define _GNU_SOURCE
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The runtime's reasons to forward (crates/vf-fwdgen/bf16tc/runtime.c, the same order). */
enum {
    R_TYPES, R_ALGO, R_BATCH, R_RANGE, R_SHAPE, R_ARGS, R_DOMAIN, R_SEQ, R_BUILD, R_K, R_HANDLE, R_CONTEXT, R_MATH,
    R_POINTER, R_ALPHA0, R_ALIAS, R_LTATTR, R_PLAN, R_CAPTURED, R_ALLOC, R_LAUNCH, R_N
};
enum { API_GEMMEX, API_GEMMEX64, API_SB, API_SB64, API_LT };

static int failures;
#define CHECK(cond, ...)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            ++failures;                                      \
            printf("FAIL  %s:%d: ", __FILE__, __LINE__);     \
            printf(__VA_ARGS__);                             \
            printf("\n");                                    \
        }                                                    \
    } while (0)
#define CU(x)                                                                         \
    do {                                                                              \
        cudaError_t e_ = (x);                                                         \
        if (e_ != cudaSuccess) {                                                      \
            printf("FAIL  %s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

/* ---- cuBLAS 12.9, directly ---- */
static struct {
    cublasStatus_t (*Create)(cublasHandle_t *);
    cublasStatus_t (*SetStream)(cublasHandle_t, cudaStream_t);
    cublasStatus_t (*SetPointerMode)(cublasHandle_t, cublasPointerMode_t);
    cublasStatus_t (*SetWorkspace)(cublasHandle_t, void *, size_t);
    __typeof__(cublasGemmEx) *GemmEx;
    __typeof__(cublasLtCreate) *LtCreate;
    __typeof__(cublasLtMatmul) *LtMatmul;
} c129;

/* The runtime's counters (test builds only; NULL with a release runtime). */
static int (*rt_counters)(int, unsigned long long *, int);

static void open_129(void)
{
    void *b = dlopen("libcublas.so.12", RTLD_NOW | RTLD_LOCAL);
    void *l = dlopen("libcublasLt.so.12", RTLD_NOW | RTLD_LOCAL);
    if (!b || !l) {
        printf("FAIL  cannot open the CUDA 12.9 libraries: %s\n", dlerror());
        exit(1);
    }
    *(void **)&c129.Create = dlsym(b, "cublasCreate_v2");
    *(void **)&c129.SetStream = dlsym(b, "cublasSetStream_v2");
    *(void **)&c129.SetPointerMode = dlsym(b, "cublasSetPointerMode_v2");
    *(void **)&c129.SetWorkspace = dlsym(b, "cublasSetWorkspace_v2");
    *(void **)&c129.GemmEx = dlsym(b, "cublasGemmEx");
    *(void **)&c129.LtCreate = dlsym(l, "cublasLtCreate");
    *(void **)&c129.LtMatmul = dlsym(l, "cublasLtMatmul");
}

/* After the first handle made through the variants (which loads the runtime). */
static void open_runtime(void)
{
    void *rt = dlopen("libvf_cublas_bf16.so", RTLD_NOW | RTLD_NOLOAD);
    if (rt) {
        *(void **)&rt_counters = dlsym(rt, "vf_bf16_test_counters");
    }
    if (!rt_counters) printf("note  a release runtime: which path served each call is not checked\n");
}

/* Counters of one entry point: [0] calls, [1] skinny, [2] tensor cores, [3] failed, [4 + r] forwarded. */
typedef struct {
    unsigned long long v[4 + R_N];
} Counts;
static Counts counts(int api)
{
    Counts c;
    memset(&c, 0, sizeof c);
    if (rt_counters && rt_counters(api, c.v, 4 + R_N) != 4 + R_N) {
        printf("FAIL  the runtime's counters do not have %d values\n", 4 + R_N);
        exit(1);
    }
    return c;
}
static unsigned long long delta(const Counts *a, const Counts *b, int i) { return b->v[i] - a->v[i]; }
#define SERVED_SKINNY 1
#define SERVED_TC 2
#define FAILED 3
#define FWD(r) (4 + (r))

/* ---- data ---- */
static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 16);
}
static double gauss(void)
{
    const double u = (rnd() + 1.0) / 4294967297.0, v = (rnd() + 1.0) / 4294967297.0;
    return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
}
static float bf2f(uint16_t b)
{
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static uint16_t f2bf(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fff;
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static uint16_t *bf16_random(size_t n, double scale)
{
    uint16_t *h = malloc(n * 2);
    for (size_t i = 0; i < n; ++i) h[i] = f2bf((float)(gauss() * scale));
    return h;
}
static void *to_device(const void *h, size_t bytes)
{
    void *d;
    CU(cudaMalloc(&d, bytes ? bytes : 1));
    CU(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
    /* A pageable copy goes through the legacy stream: finished before any call on a non-blocking
     * stream can read it. */
    CU(cudaDeviceSynchronize());
    return d;
}
static void *from_device(const void *d, size_t bytes)
{
    void *h = malloc(bytes ? bytes : 1);
    CU(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
    return h;
}
static size_t ndiff(const void *a, const void *b, size_t bytes, size_t es)
{
    size_t n = 0;
    for (size_t i = 0; i < bytes; i += es) n += memcmp((const char *)a + i, (const char *)b + i, es) != 0;
    return n;
}

/* op(X)(i, l) of a column-major bf16 matrix with leading dimension ld. */
static float opel(const uint16_t *X, int trans, long long ld, long long i, long long l)
{
    return bf2f(trans ? X[l + i * ld] : X[i + l * ld]);
}

/* The largest error of C (m x n, column-major, ldc) against an fp64 reference of alpha op(A) op(B) +
 * beta C0, in units of 2^-24 times sum |alpha a b| + |beta c|. */
static double err_units(int ta, int tb, int m, int n, int k, float alpha, const uint16_t *A, long long lda,
                        const uint16_t *B, long long ldb, float beta, const float *C0, const float *C, long long ldc)
{
    double worst = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            double s = 0, mag = 0;
            for (int l = 0; l < k; ++l) {
                const double p = (double)opel(A, ta, lda, i, l) * (double)opel(B, !tb, ldb, j, l);
                s += p;
                mag += fabs(p);
            }
            s *= alpha;
            mag *= fabs(alpha);
            if (beta != 0) {
                s += (double)beta * C0[i + (long long)j * ldc];
                mag += fabs((double)beta * C0[i + (long long)j * ldc]);
            }
            const double e = fabs((double)C[i + (long long)j * ldc] - s) / (mag > 0 ? mag * 0x1p-24 : 1.0);
            if (e > worst || isnan(e)) worst = isnan(e) ? INFINITY : e;
        }
    return worst;
}
#endif
