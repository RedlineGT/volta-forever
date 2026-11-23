/* A stand-in for libcublasLt.so.12 for the host tests of the bf16 runtime: handles; descriptors and
 * layouts that hold every attribute of CUDA 12.9's (at its size and its default), whether made by
 * Create or in the caller's space by Init; and cublasLtMatmul, which records what it was given. */
#include "stub.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EXPORT __attribute__((visibility("default")))

/* Size of each attribute (0: no such attribute) and its default, as 12.9 has them. */
static const int desc_size[40] = {4, 4, 4, 4, 4, 4, 4, 4, 8, 0, 8, 8, 8, 8, 8, 4, 0, 8, 8, 8,
                                  8, 8, 4, 8, 8, 1, 4, 4, 4, 8, 8, 4, 4, 4, 4, 4, 8, 4, 0, 0};
static const int layout_size[9] = {4, 4, 8, 8, 8, 4, 8, 8, 4};

typedef struct {
    const void *key;
    int kind; /* 0 descriptor, 1 layout */
    unsigned char v[40][8];
} Obj;
static Obj objs[2048];

static Obj *find(const void *key, int create, int kind)
{
    for (int i = 0; i < 2048; ++i)
        if (objs[i].key == key) {
            if (create) memset(objs[i].v, 0, sizeof objs[i].v);
            return &objs[i];
        }
    if (!create) return NULL;
    for (int i = 0; i < 2048; ++i)
        if (!objs[i].key) {
            objs[i].key = key;
            objs[i].kind = kind;
            memset(objs[i].v, 0, sizeof objs[i].v);
            return &objs[i];
        }
    return NULL;
}
static void put(Obj *o, int a, int64_t v)
{
    const int n = o->kind ? layout_size[a] : desc_size[a];
    memcpy(o->v[a], &v, (size_t)n); /* little endian: the low bytes */
}
static int64_t get(const Obj *o, int a)
{
    int64_t v = 0;
    const int n = o->kind ? layout_size[a] : desc_size[a];
    memcpy(&v, o->v[a], (size_t)n);
    if (n == 4) v = (int32_t)v;
    if (n == 1) v = (int8_t)v;
    return v;
}

static void desc_defaults(Obj *o, cublasComputeType_t compute, cudaDataType_t scale)
{
    put(o, CUBLASLT_MATMUL_DESC_COMPUTE_TYPE, compute);
    put(o, CUBLASLT_MATMUL_DESC_SCALE_TYPE, scale);
    put(o, CUBLASLT_MATMUL_DESC_FILL_MODE, 2); /* CUBLAS_FILL_MODE_FULL */
    put(o, CUBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_EPILOGUE_DEFAULT);
    put(o, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, -1);
}
static void layout_defaults(Obj *o, cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld)
{
    put(o, CUBLASLT_MATRIX_LAYOUT_TYPE, type);
    put(o, CUBLASLT_MATRIX_LAYOUT_ROWS, (int64_t)rows);
    put(o, CUBLASLT_MATRIX_LAYOUT_COLS, (int64_t)cols);
    put(o, CUBLASLT_MATRIX_LAYOUT_LD, ld);
    put(o, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, 1);
}

EXPORT size_t cublasLtGetVersion(void)
{
    const char *v = getenv("STUB_CUBLAS_VERSION");
    return v && *v ? (size_t)atoi(v) : 120902;
}

EXPORT cublasStatus_t cublasLtCreate(cublasLtHandle_t *h)
{
    *h = (cublasLtHandle_t)malloc(16);
    stub_event("ltcreate handle=%p", (void *)*h);
    return CUBLAS_STATUS_SUCCESS;
}
EXPORT cublasStatus_t cublasLtDestroy(cublasLtHandle_t h)
{
    free(h);
    return CUBLAS_STATUS_SUCCESS;
}
EXPORT cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *d, cublasComputeType_t c, cudaDataType_t s)
{
    *d = calloc(1, sizeof(cublasLtMatmulDescOpaque_t));
    desc_defaults(find(*d, 1, 0), c, s);
    return CUBLAS_STATUS_SUCCESS;
}
EXPORT cublasStatus_t cublasLtMatmulDescInit_internal(cublasLtMatmulDesc_t d, size_t size, cublasComputeType_t c,
                                                      cudaDataType_t s)
{
    if (size < sizeof(cublasLtMatmulDescOpaque_t)) return CUBLAS_STATUS_INVALID_VALUE;
    desc_defaults(find(d, 1, 0), c, s);
    return CUBLAS_STATUS_SUCCESS;
}
EXPORT cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t d)
{
    Obj *o = find(d, 0, 0);
    if (o) o->key = NULL;
    free(d);
    return CUBLAS_STATUS_SUCCESS;
}
static cublasStatus_t set_attr(Obj *o, int a, const void *buf, size_t size, const int *sizes, int count)
{
    if (!o || a < 0 || a >= count || !sizes[a] || size != (size_t)sizes[a]) return CUBLAS_STATUS_INVALID_VALUE;
    memcpy(o->v[a], buf, size);
    return CUBLAS_STATUS_SUCCESS;
}
static cublasStatus_t get_attr(Obj *o, int a, void *buf, size_t size, size_t *written, const int *sizes, int count)
{
    if (!o || a < 0 || a >= count || !sizes[a]) return CUBLAS_STATUS_INVALID_VALUE;
    if (size == 0 && written) {
        *written = (size_t)sizes[a];
        return CUBLAS_STATUS_SUCCESS;
    }
    if (size != (size_t)sizes[a]) return CUBLAS_STATUS_INVALID_VALUE;
    memcpy(buf, o->v[a], size);
    if (written) *written = size;
    return CUBLAS_STATUS_SUCCESS;
}
EXPORT cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t d, cublasLtMatmulDescAttributes_t a,
                                                     const void *buf, size_t size)
{
    return set_attr(find(d, 0, 0), (int)a, buf, size, desc_size, 40);
}
EXPORT cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t d, cublasLtMatmulDescAttributes_t a, void *buf,
                                                     size_t size, size_t *written)
{
    return get_attr(find(d, 0, 0), (int)a, buf, size, written, desc_size, 40);
}
EXPORT cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *l, cudaDataType t, uint64_t r, uint64_t c,
                                                 int64_t ld)
{
    *l = calloc(1, sizeof(cublasLtMatrixLayoutOpaque_t));
    layout_defaults(find(*l, 1, 1), t, r, c, ld);
    return CUBLAS_STATUS_SUCCESS;
}
EXPORT cublasStatus_t cublasLtMatrixLayoutInit_internal(cublasLtMatrixLayout_t l, size_t size, cudaDataType t,
                                                        uint64_t r, uint64_t c, int64_t ld)
{
    if (size < sizeof(cublasLtMatrixLayoutOpaque_t)) return CUBLAS_STATUS_INVALID_VALUE;
    layout_defaults(find(l, 1, 1), t, r, c, ld);
    return CUBLAS_STATUS_SUCCESS;
}
EXPORT cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t l)
{
    Obj *o = find(l, 0, 1);
    if (o) o->key = NULL;
    free(l);
    return CUBLAS_STATUS_SUCCESS;
}
EXPORT cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t l, cublasLtMatrixLayoutAttribute_t a,
                                                       const void *buf, size_t size)
{
    return set_attr(find(l, 0, 1), (int)a, buf, size, layout_size, 9);
}
EXPORT cublasStatus_t cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t l, cublasLtMatrixLayoutAttribute_t a,
                                                       void *buf, size_t size, size_t *written)
{
    return get_attr(find(l, 0, 1), (int)a, buf, size, written, layout_size, 9);
}

static void layout_text(char *out, size_t n, cublasLtMatrixLayout_t l)
{
    Obj *o = l ? find(l, 0, 1) : NULL;
    if (!o) {
        snprintf(out, n, "-");
        return;
    }
    snprintf(out, n, "%lld:%lldx%lld:ld%lld:b%lld:s%lld", (long long)get(o, 0), (long long)get(o, 2),
             (long long)get(o, 3), (long long)get(o, 4), (long long)get(o, 5), (long long)get(o, 6));
}

/* An algorithm of the stand-in: data[0] its id, data[1] its number of splits of k (host tests build them). */
EXPORT cublasStatus_t cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t *algo,
                                                          cublasLtMatmulAlgoConfigAttributes_t attr, void *buf,
                                                          size_t size, size_t *written)
{
    if (!algo || !buf || size != 4) return CUBLAS_STATUS_INVALID_VALUE;
    int32_t v;
    if (attr == CUBLASLT_ALGO_CONFIG_ID)
        v = (int32_t)algo->data[0];
    else if (attr == CUBLASLT_ALGO_CONFIG_SPLITK_NUM)
        v = (int32_t)algo->data[1];
    else
        return CUBLAS_STATUS_INVALID_VALUE;
    memcpy(buf, &v, 4);
    if (written) *written = 4;
    return CUBLAS_STATUS_SUCCESS;
}

EXPORT cublasStatus_t cublasLtMatmul(cublasLtHandle_t h, cublasLtMatmulDesc_t d, const void *alpha, const void *A,
                                     cublasLtMatrixLayout_t la, const void *B, cublasLtMatrixLayout_t lb,
                                     const void *beta, const void *C, cublasLtMatrixLayout_t lc, void *D,
                                     cublasLtMatrixLayout_t ld, const cublasLtMatmulAlgo_t *algo, void *ws,
                                     size_t wss, cudaStream_t s)
{
    Obj *o = find(d, 0, 0);
    if (!o) return CUBLAS_STATUS_INVALID_VALUE;
    if (stub_failing("cublasLtMatmul")) {
        stub_event("ltmatmul handle=%p FAILED", (void *)h);
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    char ta[96], tb[96], tc[96], td[96];
    layout_text(ta, sizeof ta, la);
    layout_text(tb, sizeof tb, lb);
    layout_text(tc, sizeof tc, lc);
    layout_text(td, sizeof td, ld);
    const int pm = (int)get(o, CUBLASLT_MATMUL_DESC_POINTER_MODE);
    stub_event("ltmatmul handle=%p compute=%lld transa=%lld transb=%lld pmode=%d epilogue=%lld alpha=%g beta=%g A=%p "
               "a=%s B=%p b=%s C=%p c=%s D=%p d=%s algo=%p ws=%zu stream=%p",
               (void *)h, (long long)get(o, 0), (long long)get(o, 3), (long long)get(o, 4), pm, (long long)get(o, 7),
               pm == 0 && alpha ? *(const float *)alpha : -999.0, pm == 0 && beta ? *(const float *)beta : -999.0, A,
               ta, B, tb, C, tc, D, td, (const void *)algo, wss, (void *)s);
    return CUBLAS_STATUS_SUCCESS;
}
