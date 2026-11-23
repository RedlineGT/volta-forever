/* The bf16 cuBLAS variants on a V100: which calls each path serves, what it computes against fp64 and
 * against cuBLAS 12.9, and what is forwarded (review section 5, G9). Prints "ok bf16-paths" when every
 * check passes. Run by tests/gpu/forward-bf16.sh with VF_CUBLAS_BF16_TC=1 and a test build of the
 * runtime (its counters say which path served a call). */
#include "bf16-common.h"

typedef struct {
    int ta, tb, m, n, k;
    cudaDataType ctype;
    long long lda, ldb, ldc;
} Shape;

static Shape shape(int ta, int tb, int m, int n, int k, cudaDataType ct)
{
    Shape s = {ta, tb, m, n, k, ct, ta ? k : m, tb ? n : k, m};
    return s;
}

typedef struct {
    uint16_t *hA, *hB;
    void *dA, *dB;
    float *hC0; /* initial C as fp32 (bf16 C: its values) */
    void *hCraw;
    size_t cbytes;
} Data;

static Data make_data(const Shape *s, double sa, double sb)
{
    Data d;
    const size_t na = (size_t)s->lda * (s->ta ? s->m : s->k), nb = (size_t)s->ldb * (s->tb ? s->k : s->n);
    d.hA = bf16_random(na, sa);
    d.hB = bf16_random(nb, sb);
    d.dA = to_device(d.hA, na * 2);
    d.dB = to_device(d.hB, nb * 2);
    const size_t nc = (size_t)s->ldc * s->n;
    const size_t es = s->ctype == CUDA_R_32F ? 4 : 2;
    d.cbytes = nc * es;
    d.hC0 = malloc(nc * 4);
    d.hCraw = malloc(d.cbytes);
    for (size_t i = 0; i < nc; ++i) {
        const float v = (float)gauss();
        if (es == 4) {
            ((float *)d.hCraw)[i] = v;
            d.hC0[i] = v;
        } else {
            ((uint16_t *)d.hCraw)[i] = f2bf(v);
            d.hC0[i] = bf2f(f2bf(v));
        }
    }
    return d;
}

static void free_data(Data *d)
{
    cudaFree(d->dA);
    cudaFree(d->dB);
    free(d->hA);
    free(d->hB);
    free(d->hC0);
    free(d->hCraw);
}

static cublasHandle_t hv, h12; /* through the variant; 12.9 directly */
static cudaStream_t stream;

/* One call through the variant and the same through 12.9; returns both outputs (host copies). */
static void both(const Shape *s, const Data *d, float alpha, float beta, cublasGemmAlgo_t algo, void **outv,
                 void **out12, cublasStatus_t *stv, cublasStatus_t *st12)
{
    void *Cv = to_device(d->hCraw, d->cbytes), *C12 = to_device(d->hCraw, d->cbytes);
    *stv = cublasGemmEx(hv, s->ta ? CUBLAS_OP_T : CUBLAS_OP_N, s->tb ? CUBLAS_OP_T : CUBLAS_OP_N, s->m, s->n, s->k,
                        &alpha, d->dA, CUDA_R_16BF, (int)s->lda, d->dB, CUDA_R_16BF, (int)s->ldb, &beta, Cv, s->ctype,
                        (int)s->ldc, CUBLAS_COMPUTE_32F, algo);
    *st12 = c129.GemmEx(h12, s->ta ? CUBLAS_OP_T : CUBLAS_OP_N, s->tb ? CUBLAS_OP_T : CUBLAS_OP_N, s->m, s->n, s->k,
                        &alpha, d->dA, CUDA_R_16BF, (int)s->lda, d->dB, CUDA_R_16BF, (int)s->ldb, &beta, C12,
                        s->ctype, (int)s->ldc, CUBLAS_COMPUTE_32F, algo);
    CU(cudaStreamSynchronize(stream));
    CU(cudaDeviceSynchronize());
    *outv = from_device(Cv, d->cbytes);
    *out12 = from_device(C12, d->cbytes);
    cudaFree(Cv);
    cudaFree(C12);
}

static float *as_f32(const Shape *s, const void *raw)
{
    const size_t nc = (size_t)s->ldc * s->n;
    float *f = malloc(nc * 4);
    for (size_t i = 0; i < nc; ++i)
        f[i] = s->ctype == CUDA_R_32F ? ((const float *)raw)[i] : bf2f(((const uint16_t *)raw)[i]);
    return f;
}

static const char *layout_name(const Shape *s)
{
    static const char *n[4] = {"NN", "NT", "TN", "TT"};
    return n[s->ta * 2 + s->tb];
}

/* The skinny path: served, and bit-identical to 12.9 where 12.9 runs its sequential kernel (the
 * layouts and shapes of the kernel sweep, results/sweep2: NT and TT at every size, NN from m = 448). */
static void skinny_cases(void)
{
    const int ns[] = {2, 5, 8, 13, 16};
    const int layouts[][2] = {{0, 1}, {1, 1}, {0, 0}};
    for (int L = 0; L < 3; ++L)
        for (int i = 0; i < 5; ++i)
            for (int ct = 0; ct < 2; ++ct) {
                Shape s = shape(layouts[L][0], layouts[L][1], 1536, ns[i], 1920, ct ? CUDA_R_16BF : CUDA_R_32F);
                Data d = make_data(&s, 0.05, 1.0);
                void *ov, *o12;
                cublasStatus_t a, b;
                const Counts c0 = counts(API_GEMMEX);
                both(&s, &d, 1.25f, 0.5f, CUBLAS_GEMM_DEFAULT_TENSOR_OP, &ov, &o12, &a, &b);
                const Counts c1 = counts(API_GEMMEX);
                const size_t diff = ndiff(ov, o12, d.cbytes, s.ctype == CUDA_R_32F ? 4 : 2);
                CHECK(a == 0 && b == 0, "skinny %s n=%d: status %d / %d", layout_name(&s), s.n, a, b);
                CHECK(!rt_counters || delta(&c0, &c1, SERVED_SKINNY) == 1, "skinny %s n=%d: not served by the skinny path",
                      layout_name(&s), s.n);
                CHECK(diff == 0, "skinny %s n=%d %s: %zu elements differ from 12.9", layout_name(&s), s.n,
                      ct ? "bf16" : "fp32", diff);
                free(ov);
                free(o12);
                free_data(&d);
            }
}

/* The tensor-core path: served, within 32 units of fp64 (2^-24 sum |a b|), deterministic; the same
 * call captured in a graph goes to 12.9 (its workspace would be a node of the graph): the replay has
 * 12.9's bits. */
static void tc_cases(void)
{
    const int layouts[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    for (int L = 0; L < 4; ++L)
        for (int ct = 0; ct < 2; ++ct) {
            Shape s = shape(layouts[L][0], layouts[L][1], 520, 72, 2051, ct ? CUDA_R_16BF : CUDA_R_32F);
            Data d = make_data(&s, 0.05, 1.0);
            void *ov, *o12, *ov2, *o122;
            cublasStatus_t a, b;
            const Counts c0 = counts(API_GEMMEX);
            both(&s, &d, 1.0f, 0.0f, CUBLAS_GEMM_DEFAULT, &ov, &o12, &a, &b);
            both(&s, &d, 1.0f, 0.0f, CUBLAS_GEMM_DEFAULT, &ov2, &o122, &a, &b);
            const Counts c1 = counts(API_GEMMEX);
            CHECK(!rt_counters || delta(&c0, &c1, SERVED_TC) == 2, "tc %s: not served by the tensor-core path",
                  layout_name(&s));
            const size_t es = s.ctype == CUDA_R_32F ? 4 : 2;
            CHECK(ndiff(ov, ov2, d.cbytes, es) == 0, "tc %s: two runs differ", layout_name(&s));
            float *f = as_f32(&s, ov);
            const double e = err_units(s.ta, s.tb, s.m, s.n, s.k, 1.0f, d.hA, s.lda, d.hB, s.ldb, 0.0f, d.hC0, f, s.ldc);
            /* fp32 output: the arithmetic's own bound; bf16 output adds half an ulp of bf16 (2^-9 relative). */
            CHECK(e <= (ct ? 32.0 + 32768.0 : 32.0), "tc %s %s: error %.2f units", layout_name(&s), ct ? "bf16" : "fp32", e);
            /* captured: 12.9's */
            void *Cg = to_device(d.hCraw, d.cbytes);
            const Counts k0 = counts(API_GEMMEX);
            cudaGraph_t g;
            cudaGraphExec_t ge;
            const float one = 1.0f, zero = 0.0f;
            CU(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            cublasGemmEx(hv, s.ta ? CUBLAS_OP_T : CUBLAS_OP_N, s.tb ? CUBLAS_OP_T : CUBLAS_OP_N, s.m, s.n, s.k, &one,
                         d.dA, CUDA_R_16BF, (int)s.lda, d.dB, CUDA_R_16BF, (int)s.ldb, &zero, Cg, s.ctype, (int)s.ldc,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            CU(cudaStreamEndCapture(stream, &g));
            const Counts k1 = counts(API_GEMMEX);
            CHECK(!rt_counters || delta(&k0, &k1, FWD(R_CAPTURED)) == 1, "tc %s: a captured call not forwarded",
                  layout_name(&s));
            CU(cudaGraphInstantiate(&ge, g, 0));
            for (int r = 0; r < 3; ++r) CU(cudaGraphLaunch(ge, stream));
            CU(cudaStreamSynchronize(stream));
            void *og = from_device(Cg, d.cbytes);
            CHECK(ndiff(o12, og, d.cbytes, es) == 0, "tc %s: the replay of the captured call is not 12.9's", layout_name(&s));
            cudaGraphExecDestroy(ge);
            cudaGraphDestroy(g);
            cudaFree(Cg);
            free(og);
            free(f);
            free(ov);
            free(o12);
            free(ov2);
            free(o122);
            free_data(&d);
        }
}

/* Forwarded calls: 12.9's bits and status, and the reason counted. */
static void forwarded_case(const char *what, Shape s, float alpha, cublasGemmAlgo_t algo, int reason)
{
    Data d = make_data(&s, 0.05, 1.0);
    void *ov, *o12;
    cublasStatus_t a, b;
    const Counts c0 = counts(API_GEMMEX);
    both(&s, &d, alpha, 0.5f, algo, &ov, &o12, &a, &b);
    const Counts c1 = counts(API_GEMMEX);
    CHECK(a == b, "%s: status %d, 12.9 gives %d", what, a, b);
    CHECK(ndiff(ov, o12, d.cbytes, s.ctype == CUDA_R_32F ? 4 : 2) == 0, "%s: not 12.9's bits", what);
    CHECK(!rt_counters || delta(&c0, &c1, FWD(reason)) == 1, "%s: not forwarded for the expected reason", what);
    free(ov);
    free(o12);
    free_data(&d);
}

static void forwarded_cases(void)
{
    forwarded_case("explicit algorithm ALGO5", shape(0, 0, 512, 64, 512, CUDA_R_32F), 1.0f, CUBLAS_GEMM_ALGO5, R_ALGO);
    forwarded_case("explicit algorithm ALGO3_TENSOR_OP", shape(1, 0, 512, 64, 512, CUDA_R_32F), 1.0f,
                   CUBLAS_GEMM_ALGO3_TENSOR_OP, R_ALGO);
    forwarded_case("k = 16392 on the tensor-core path", shape(1, 0, 64, 40, 16392, CUDA_R_32F), 1.0f,
                   CUBLAS_GEMM_DEFAULT, R_K);
    forwarded_case("n = 1", shape(1, 0, 512, 1, 512, CUDA_R_32F), 1.0f, CUBLAS_GEMM_DEFAULT, R_SHAPE);
    forwarded_case("k = 8", shape(1, 0, 512, 64, 8, CUDA_R_32F), 1.0f, CUBLAS_GEMM_DEFAULT, R_SHAPE);
    forwarded_case("alpha = 0", shape(1, 0, 512, 64, 2048, CUDA_R_32F), 0.0f, CUBLAS_GEMM_DEFAULT, R_ALPHA0);
    forwarded_case("k = 2040, under the tensor-core path's gate", shape(1, 0, 512, 64, 2040, CUDA_R_32F), 1.0f,
                   CUBLAS_GEMM_DEFAULT, R_SHORT_K);
    forwarded_case("small product (m k < 2^20, n < 32, outside the skinny domain)", shape(0, 1, 16384, 20, 16, CUDA_R_32F),
                   1.0f, CUBLAS_GEMM_DEFAULT, R_DOMAIN);
}

/* The skinny path serves only where 12.9 runs its sequential kernel (results of the kernel capture,
 * tests/gpu/forward/bf16-skinny.c): TN with m = 384 is where 12.9 runs gemmSN, and a workspace under
 * 4 MiB changes 12.9's choice for op(B) = N; cublasSetStream gives the handle its default workspace
 * back. Run with the skinny path alone (VF_CUBLAS_BF16_TC=0 VF_CUBLAS_BF16_SKINNY=1), so that such
 * calls are forwarded rather than served by the tensor-core path. */
static void sequential_rule_cases(void)
{
    forwarded_case("TN m = 384 (12.9 runs gemmSN there)", shape(1, 0, 384, 8, 1536, CUDA_R_32F), 1.0f,
                   CUBLAS_GEMM_DEFAULT, R_SEQ);
    forwarded_case("TN m = 777 (not a multiple of 8)", shape(1, 0, 777, 8, 1536, CUDA_R_32F), 1.0f, CUBLAS_GEMM_DEFAULT,
                   R_SEQ);
    void *ws;
    CU(cudaMalloc(&ws, 1u << 20));
    cublasSetWorkspace(hv, ws, 1u << 20);
    forwarded_case("TN m = 1536 with a 1 MiB workspace", shape(1, 0, 1536, 8, 1536, CUDA_R_32F), 1.0f,
                   CUBLAS_GEMM_DEFAULT, R_SEQ);
    cublasSetStream(hv, stream); /* the default workspace again */
    Shape s = shape(1, 0, 1536, 8, 1536, CUDA_R_32F);
    Data d = make_data(&s, 0.05, 1.0);
    void *ov, *o12;
    cublasStatus_t a, b;
    const Counts c0 = counts(API_GEMMEX);
    both(&s, &d, 1.0f, 0.0f, CUBLAS_GEMM_DEFAULT, &ov, &o12, &a, &b);
    const Counts c1 = counts(API_GEMMEX);
    CHECK(!rt_counters || delta(&c0, &c1, SERVED_SKINNY) == 1, "TN m = 1536 after cublasSetStream: not served");
    CHECK(ndiff(ov, o12, d.cbytes, 4) == 0, "TN m = 1536 after cublasSetStream: not 12.9's bits");
    free(ov);
    free(o12);
    free_data(&d);
    cudaFree(ws);
}

/* beta = 0 never reads C: NaN in C does not reach the output (both paths). */
static void beta_zero_case(int n)
{
    Shape s = shape(1, 0, 768, n, 2048, CUDA_R_32F);
    Data d = make_data(&s, 0.05, 1.0);
    for (size_t i = 0; i < (size_t)s.ldc * s.n; ++i) ((float *)d.hCraw)[i] = NAN;
    void *ov, *o12;
    cublasStatus_t a, b;
    both(&s, &d, 1.0f, 0.0f, CUBLAS_GEMM_DEFAULT, &ov, &o12, &a, &b);
    size_t nans = 0;
    for (size_t i = 0; i < (size_t)s.ldc * s.n; ++i) nans += isnan(((float *)ov)[i]);
    CHECK(nans == 0, "beta = 0, n = %d: %zu NaN from C in the output", n, nans);
    free(ov);
    free(o12);
    free_data(&d);
}

/* Rows and columns that fail the check (infinities, NaN, values out of the window) take the
 * fallback; C in place with beta != 0. Their elements are 12.9's where 12.9 runs its sequential
 * kernel (NT); the others stay within the bound. */
static void failing_case(void)
{
    Shape s = shape(0, 1, 640, 96, 2048, CUDA_R_32F);
    Data d = make_data(&s, 0.05, 1.0);
    const int bad_rows[] = {3, 77, 500};
    for (int r = 0; r < 3; ++r) d.hA[bad_rows[r] + (size_t)200 * s.lda] = f2bf(r == 0 ? INFINITY : r == 1 ? 3e38f : 1e-30f);
    const int bad_col = 17; /* op(B) = B^T: column j of op(B) is row j of B */
    d.hB[bad_col + (size_t)5 * s.ldb] = f2bf(-INFINITY);
    CU(cudaMemcpy(d.dA, d.hA, (size_t)s.lda * s.k * 2, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d.dB, d.hB, (size_t)s.ldb * s.k * 2, cudaMemcpyHostToDevice));
    void *ov, *o12;
    cublasStatus_t a, b;
    const Counts c0 = counts(API_GEMMEX);
    both(&s, &d, 0.75f, 0.5f, CUBLAS_GEMM_DEFAULT, &ov, &o12, &a, &b);
    const Counts c1 = counts(API_GEMMEX);
    CHECK(!rt_counters || delta(&c0, &c1, SERVED_TC) == 1, "failing rows: not served by the tensor-core path");
    const float *fv = ov, *f12 = o12;
    size_t bad = 0, checked = 0;
    for (int j = 0; j < s.n; ++j)
        for (int i = 0; i < s.m; ++i) {
            const int fails = i == 3 || i == 77 || i == 500 || j == bad_col;
            if (!fails) continue;
            ++checked;
            if (memcmp(&fv[i + (size_t)j * s.ldc], &f12[i + (size_t)j * s.ldc], 4)) ++bad;
        }
    CHECK(checked > 0 && bad == 0, "failing rows and column: %zu of %zu fallback elements differ from 12.9", bad, checked);
    free(ov);
    free(o12);
    free_data(&d);
}

/* Device pointer mode: alpha and beta read when the kernels run, so a graph replay uses the scalars
 * of that moment. */
static void device_scalars_case(void)
{
    Shape s = shape(1, 0, 512, 64, 2048, CUDA_R_32F);
    Data d = make_data(&s, 0.05, 1.0);
    float *dab;
    CU(cudaMalloc((void **)&dab, 8));
    void *C = to_device(d.hCraw, d.cbytes);
    cublasSetPointerMode(hv, CUBLAS_POINTER_MODE_DEVICE);
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CU(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    cublasGemmEx(hv, CUBLAS_OP_T, CUBLAS_OP_N, s.m, s.n, s.k, dab, d.dA, CUDA_R_16BF, (int)s.lda, d.dB, CUDA_R_16BF,
                 (int)s.ldb, dab + 1, C, CUDA_R_32F, (int)s.ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CU(cudaStreamEndCapture(stream, &g));
    cublasSetPointerMode(hv, CUBLAS_POINTER_MODE_HOST);
    CU(cudaGraphInstantiate(&ge, g, 0));
    float *out[2];
    const float scal[2][2] = {{1.0f, 0.0f}, {2.0f, 0.0f}};
    for (int r = 0; r < 2; ++r) {
        CU(cudaMemcpy(dab, scal[r], 8, cudaMemcpyHostToDevice));
        CU(cudaGraphLaunch(ge, stream));
        CU(cudaStreamSynchronize(stream));
        out[r] = from_device(C, d.cbytes);
    }
    size_t bad = 0;
    for (size_t i = 0; i < (size_t)s.ldc * s.n; ++i) bad += out[1][i] != 2.0f * out[0][i];
    CHECK(bad == 0, "device scalars: %zu elements do not follow alpha changed between replays", bad);
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    cudaFree(C);
    cudaFree(dab);
    free(out[0]);
    free(out[1]);
    free_data(&d);
}

int main(void)
{
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    CU(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (cublasCreate(&hv) || c129.Create(&h12)) {
        printf("FAIL  cannot create the handles\n");
        return 1;
    }
    cublasSetStream(hv, stream);
    c129.SetStream(h12, stream);
    open_runtime();
    const char *tcv = getenv("VF_CUBLAS_BF16_TC");
    const int tc_on = tcv && !strcmp(tcv, "1");
    skinny_cases();
    beta_zero_case(8);
    if (!tc_on) { /* the skinny path alone: what it forwards */
        sequential_rule_cases();
        if (failures) {
            printf("%d checks failed\n", failures);
            return 1;
        }
        printf("ok bf16-paths\n");
        return 0;
    }
    tc_cases();
    forwarded_cases();
    beta_zero_case(64);
    failing_case();
    /* (The control that runs the internal GEMM through the caller's handle would read the host's 1.0
     * as a device address in device pointer mode, a fault that the review predicted: not run with it.) */
    const char *control = getenv("VF_BF16_CONTROL");
    if (!control || !strstr(control, "caller-handle")) device_scalars_case();
    if (failures) {
        printf("%d checks failed\n", failures);
        return 1;
    }
    printf("ok bf16-paths\n");
    return 0;
}
