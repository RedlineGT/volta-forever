/* The time of one bf16 product through the variants and through cuBLAS 12.9 directly, per layout, shape and kind of
 * data, with the path that served it (a test build of the runtime: its counters). The measurement behind the
 * tensor-core path's layout and tall rules (docs/performance.md, section 7); run by hand on a V100, with the test
 * runtime and VF_CUBLAS_BF16_TC=1 (to time the path where a rule forwards, the test build's controls
 * VF_BF16_CONTROL=tc-layouts,tc-tall remove the rules; not part of tests/gpu/forward-bf16.sh):
 *   bf16-layout-cost OUT.tsv SET
 * SET is train (the products of a fine-tuning step of a model of hidden size 896, MLP size 4864, vocabulary 151936,
 * rank-16 adapters, at 2048 and 4096 tokens, with each kind of data), grid (the four layouts, m 128 to 4096, n 32 to
 * 2048, k 2048 to 8192), sweep (TN, m 4096 to 262144 at k = 2048, 3072, 4096, and to 65536 at k = 8192, 12288, 16384,
 * n = 2 to 512), or a file with one product per line, "LAYOUT m n k DATA" ('#' starts a comment). LAYOUT is one of
 * NN NT TN TT (cuBLAS's op(A) op(B)); DATA is ok (A ~ N(0, 0.05), B ~ N(0, 1): every row and column converts exactly),
 * fa (one element of 1e-30 in every row of op(A): every row fails the exact-fp16 check), fb (the same in every column
 * of op(B)), fab (both), hb (every other column of op(B)) or fNN (NN % of the columns of op(B), spread evenly).
 * The GPU's clock is brought up first (12.9's fp16 products for VF_WARM seconds, default 12). Each product: one call
 * through the variants (its path from the counters), then 3 untimed calls of each, then alternately one timed call of
 * each (CUDA events), 7 times; the medians are printed, and 12.9's time over the variant's. fp32 output, alpha 1,
 * beta 0, a non-blocking stream. */
#include "bf16-common.h"
#include <time.h>

static int cmpf(const void *a, const void *b)
{
    const float x = *(const float *)a, y = *(const float *)b;
    return x < y ? -1 : x > y;
}

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static cublasHandle_t hv, h12;
static cudaStream_t st;

static cublasStatus_t gemm(int variant, int ta, int tb, int m, int n, int k, const void *A, long long lda,
                           const void *B, long long ldb, void *C)
{
    const float one = 1.0f, zero = 0.0f;
    return (variant ? cublasGemmEx : c129.GemmEx)(variant ? hv : h12, ta ? CUBLAS_OP_T : CUBLAS_OP_N,
                                                  tb ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &one, A, CUDA_R_16BF,
                                                  (int)lda, B, CUDA_R_16BF, (int)ldb, &zero, C, CUDA_R_32F, m,
                                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

static const char *reason_short(const Counts *a, const Counts *b)
{
    static const char *nm[R_N] = {"types",   "algo",   "batch",   "range", "shape",   "args",    "domain", "seq",
                                  "build",   "k>16384", "k<gate", "safe", "layout", "tall", "wide", "handle",  "context", "math",
                                  "pointer", "alpha0", "alias",   "ltattr", "plan",  "captured", "alloc",  "launch"};
    if (delta(a, b, SERVED_TC)) return "tc";
    if (delta(a, b, SERVED_SKINNY)) return "skinny";
    for (int r = 0; r < R_N; ++r)
        if (delta(a, b, FWD(r))) return nm[r];
    return "fwd?"; /* a reason past the ones this tool knows */
}

/* The built-in sets, written to a temporary file in the format of a file of products. */
static FILE *builtin(const char *name)
{
    static const int ns[] = {2, 4, 8, 12, 16, 17, 24, 32, 48, 64, 65, 72, 80, 96, 128, 192, 256, 384, 512};
    static const int ms[] = {4096,  6144,  7168,  8192,  9216,   10240,  12288,  14336,  16384, 20480,
                             24576, 32768, 49152, 65536, 98304, 131072, 151936, 196608, 262144};
    static const char *const lays[] = {"NN", "NT", "TT", "TN"};
    static const struct {
        const char *lay;
        int m, n, k; /* n or k of 0: the tokens */
    } train[] = {{"TN", 896, 0, 4864},   {"TN", 16, 0, 4864},   {"NN", 896, 0, 4864},  {"NN", 16, 0, 4864},
                 {"NT", 16, 4864, 0},    {"NT", 4864, 16, 0},   {"NT", 16, 896, 0},    {"NT", 16, 128, 0},
                 {"NT", 896, 16, 0},     {"NT", 896, 4864, 0},  {"NT", 896, 151936, 0}, {"NT", 4864, 896, 0},
                 {"NT", 896, 896, 0},    {"NT", 896, 128, 0}};
    FILE *f;
    if (!strcmp(name, "train")) {
        f = tmpfile();
        for (int t = 2048; t <= 4096; t *= 2)
            for (size_t p = 0; p < sizeof train / sizeof *train; ++p)
                for (int d = 0; d < 3; ++d)
                    fprintf(f, "%s %d %d %d %s\n", train[p].lay, train[p].m, train[p].n ? train[p].n : t,
                            train[p].k ? train[p].k : t, d == 0 ? "ok" : d == 1 ? "fb" : "fa");
    } else if (!strcmp(name, "grid")) {
        f = tmpfile();
        static const int gm[] = {128, 512, 1024, 2048, 4096}, gn[] = {32, 128, 512, 2048}, gk[] = {2048, 4096, 8192};
        for (int l = 0; l < 4; ++l)
            for (int a = 0; a < 5; ++a)
                for (int b = 0; b < 4; ++b)
                    for (int c = 0; c < 3; ++c) fprintf(f, "%s %d %d %d ok\n", lays[l], gm[a], gn[b], gk[c]);
    } else if (!strcmp(name, "sweep")) {
        f = tmpfile();
        static const int ks[] = {2048, 3072, 4096, 8192, 12288, 16384};
        for (int c = 0; c < 6; ++c)
            for (size_t a = 0; a < sizeof ms / sizeof *ms; ++a) {
                if (ks[c] > 4096 && ms[a] > 65536) continue;
                for (size_t b = 0; b < sizeof ns / sizeof *ns; ++b) fprintf(f, "TN %d %d %d ok\n", ms[a], ns[b], ks[c]);
            }
    } else {
        return NULL;
    }
    rewind(f);
    return f;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s OUT.tsv train|grid|sweep|FILE\n", argv[0]);
        return 2;
    }
    FILE *out = fopen(argv[1], "w"), *in = builtin(argv[2]);
    if (!in) in = fopen(argv[2], "r");
    if (!out || !in) {
        perror("open");
        return 2;
    }
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    CU(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    if (cublasCreate(&hv) || c129.Create(&h12)) return 1;
    open_runtime();
    cublasSetStream(hv, st);
    c129.SetStream(h12, st);
    {   /* the clock up: 12.9's fp16 products, 4096 x 4096 x 4096 */
        const double warm = getenv("VF_WARM") ? atof(getenv("VF_WARM")) : 12.0;
        void *X;
        CU(cudaMalloc(&X, 3ull * 4096 * 4096 * 2));
        CU(cudaMemset(X, 0, 3ull * 4096 * 4096 * 2));
        const float one = 1.0f, zero = 0.0f;
        const double t0 = now_s();
        while (now_s() - t0 < warm) {
            for (int i = 0; i < 10; ++i)
                c129.GemmEx(h12, CUBLAS_OP_N, CUBLAS_OP_N, 4096, 4096, 4096, &one, X, CUDA_R_16F, 4096,
                            (char *)X + 4096ull * 4096 * 2, CUDA_R_16F, 4096, &zero, (char *)X + 2 * 4096ull * 4096 * 2,
                            CUDA_R_16F, 4096, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            CU(cudaStreamSynchronize(st));
        }
        cudaFree(X);
    }
    fprintf(out, "layout\tm\tn\tk\tdata\tpath\tus_variant\tus_129\tratio_129_over_variant\n");
    fflush(out);
    char line[256];
    while (fgets(line, sizeof line, in)) {
        char lay[8], data[16];
        int m, n, k;
        if (line[0] == '#' || sscanf(line, "%7s %d %d %d %15s", lay, &m, &n, &k, data) != 5) continue;
        const int ta = lay[0] == 'T', tb = lay[1] == 'T';
        const long long lda = ta ? k : m, ldb = tb ? n : k;
        const size_t na = (size_t)lda * (ta ? m : k), nb = (size_t)ldb * (tb ? k : n);
        const uint16_t tiny = f2bf(1e-30f);
        const int fa = !strcmp(data, "fa") || !strcmp(data, "fab"), fb = !strcmp(data, "fb") || !strcmp(data, "fab"),
                  hb = !strcmp(data, "hb");
        /* A is kept from the previous product when its layout, m, k and kind are the same (only n changed) */
        static void *A = NULL;
        static int pa_ta = -1, pa_m, pa_k, pa_fa;
        if (!(A && pa_ta == ta && pa_m == m && pa_k == k && pa_fa == fa)) {
            if (A) cudaFree(A);
            uint16_t *hA = malloc(na * 2);
            for (size_t i = 0; i < na; ++i) hA[i] = f2bf((float)(gauss() * 0.05));
            if (fa)
                for (int i = 0; i < m; ++i) { /* op(A)(i, l) at l = i % k */
                    const long long l = i % k;
                    hA[ta ? l + (size_t)i * lda : i + (size_t)l * lda] = tiny;
                }
            A = to_device(hA, na * 2);
            free(hA);
            pa_ta = ta, pa_m = m, pa_k = k, pa_fa = fa;
        }
        uint16_t *hB = malloc(nb * 2);
        for (size_t i = 0; i < nb; ++i) hB[i] = f2bf((float)gauss());
        /* fNN: NN % of the columns of op(B) fail, spread evenly (column j fails when floor((j + 1) p) > floor(j p)) */
        const double frac = fb   ? 1.0
                            : hb ? 0.5
                                 : (data[0] == 'f' && data[1] >= '0' && data[1] <= '9') ? atof(data + 1) / 100.0 : 0.0;
        for (int j = 0; j < n; ++j)
            if (frac > 0 && floor((j + 1) * frac) > floor(j * frac)) { /* op(B)(l, j) at l = j % k */
                const long long l = j % k;
                hB[tb ? j + (size_t)l * ldb : l + (size_t)j * ldb] = tiny;
            }
        void *B = to_device(hB, nb * 2), *C;
        free(hB);
        CU(cudaMalloc(&C, (size_t)m * n * 4));
        const Counts c0 = counts(API_GEMMEX);
        const cublasStatus_t s1 = gemm(1, ta, tb, m, n, k, A, lda, B, ldb, C);
        CU(cudaStreamSynchronize(st));
        const Counts c1 = counts(API_GEMMEX);
        const char *path = rt_counters ? reason_short(&c0, &c1) : "?";
        for (int r = 0; r < 3; ++r) {
            gemm(1, ta, tb, m, n, k, A, lda, B, ldb, C);
            gemm(0, ta, tb, m, n, k, A, lda, B, ldb, C);
        }
        CU(cudaStreamSynchronize(st));
        float tv[7], t12[7];
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        for (int r = 0; r < 7; ++r)
            for (int v = 1; v >= 0; --v) {
                cudaEventRecord(e0, st);
                gemm(v, ta, tb, m, n, k, A, lda, B, ldb, C);
                cudaEventRecord(e1, st);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(v ? &tv[r] : &t12[r], e0, e1);
            }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        qsort(tv, 7, sizeof(float), cmpf);
        qsort(t12, 7, sizeof(float), cmpf);
        const double uv = tv[3] * 1e3, u12 = t12[3] * 1e3;
        fprintf(out, "%s\t%d\t%d\t%d\t%s\t%s%s\t%.1f\t%.1f\t%.3f\n", lay, m, n, k, data, path,
                s1 ? "(status!)" : "", uv, u12, u12 / uv);
        fflush(out);
        printf("%s %6d %6d %6d %-3s %-8s variant %9.1f us  12.9 %9.1f us  12.9/variant %.3f\n", lay, m, n, k, data,
               path, uv, u12, u12 / uv);
        fflush(stdout);
        cudaFree(B);
        cudaFree(C);
    }
    fclose(out);
    return failures != 0; /* bf16-common.h's CHECK, should a helper use it */
}
