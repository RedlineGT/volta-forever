/* What the figures of INSTALL.md, LIMITS.md and docs/emulations.md (section 11) about the tensor-core
 * path's cost and error say, measured on this implementation (a measurement, run by hand on a V100
 * with the release runtime; not part of tests/gpu/forward-bf16.sh):
 *   1. host time of one eager call: tensor-core path (TN 1536 x 480 x 1536), skinny path (TN
 *      1536 x 8 x 1536), and the same calls given to cuBLAS 12.9 directly (mean over 100 calls);
 *   2. the tensor-core path with every column of op(B) failing the strict check (one element of
 *      1e-30 in each), against the same call with no failure and against 12.9, at n = 480, for six
 *      shapes of linear layers (median of 5, CUDA events);
 *   3. same-sign data (bf16 uniform in [0.5, 1)): the largest error and the mean signed error over
 *      256 sampled elements, in units of 2^-24 sum |a b| against fp64, for three shapes.
 * Prints its tables. */
#include "bf16-common.h"
#include <time.h>

static cublasHandle_t hv, h12;
static cudaStream_t st;

static double now_us(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static void gemm(int variant, int m, int n, int k, const void *A, const void *B, void *C)
{
    const float one = 1.0f, zero = 0.0f;
    if (variant)
        cublasGemmEx(hv, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, A, CUDA_R_16BF, k, B, CUDA_R_16BF, k, &zero, C,
                     CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    else
        c129.GemmEx(h12, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, A, CUDA_R_16BF, k, B, CUDA_R_16BF, k, &zero, C,
                    CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

static float gpu_us(int variant, int m, int n, int k, const void *A, const void *B, void *C)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    float t[5];
    gemm(variant, m, n, k, A, B, C);
    for (int r = 0; r < 5; ++r) {
        cudaEventRecord(a, st);
        gemm(variant, m, n, k, A, B, C);
        cudaEventRecord(b, st);
        cudaEventSynchronize(b);
        cudaEventElapsedTime(&t[r], a, b);
    }
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j)
            if (t[j] < t[i]) {
                const float x = t[i];
                t[i] = t[j];
                t[j] = x;
            }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return t[2] * 1000.0f;
}

int main(void)
{
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    CU(cudaStreamCreate(&st));
    if (cublasCreate(&hv) || c129.Create(&h12)) return 1;
    open_runtime();
    cublasSetStream(hv, st);
    c129.SetStream(h12, st);

    /* 1. host time per call */
    {
        const int m = 1536, k = 1536;
        uint16_t *hA = bf16_random((size_t)m * k, 0.05), *hB = bf16_random((size_t)480 * k, 1.0);
        void *A = to_device(hA, (size_t)m * k * 2), *B = to_device(hB, (size_t)480 * k * 2), *C;
        CU(cudaMalloc(&C, (size_t)m * 480 * 4));
        static const int ns[] = {480, 8};
        for (int i = 0; i < 2; ++i)
            for (int v = 1; v >= 0; --v) {
                for (int w = 0; w < 10; ++w) gemm(v, m, ns[i], k, A, B, C);
                CU(cudaStreamSynchronize(st));
                const double t0 = now_us();
                for (int r = 0; r < 100; ++r) gemm(v, m, ns[i], k, A, B, C);
                const double t1 = now_us();
                CU(cudaStreamSynchronize(st));
                printf("host time per call, TN %d x %d x %d, %s: %.1f us\n", m, ns[i], k,
                       v ? (ns[i] >= 32 ? "variant (tensor-core path)" : "variant (skinny path)") : "cuBLAS 12.9",
                       (t1 - t0) / 100);
            }
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
        free(hA);
        free(hB);
    }

    /* 2. every column failing, n = 480 */
    static const int shapes[][2] = {{1536, 1536}, {4608, 1536}, {1536, 4608}, {7344, 1536}, {9728, 896}, {896, 4864}};
    printf("%-14s %10s %14s %10s %14s\n", "m x k (n=480)", "tc us", "all fail us", "12.9 us", "all fail / 12.9");
    for (size_t s = 0; s < sizeof shapes / sizeof *shapes; ++s) {
        const int m = shapes[s][0], k = shapes[s][1], n = 480;
        uint16_t *hA = bf16_random((size_t)m * k, 0.05), *hB = bf16_random((size_t)n * k, 1.0);
        void *A = to_device(hA, (size_t)m * k * 2), *B = to_device(hB, (size_t)n * k * 2), *C;
        for (int j = 0; j < n; ++j) hB[(size_t)(j % k) + (size_t)j * k] = f2bf(1e-30f);
        void *Bf = to_device(hB, (size_t)n * k * 2);
        CU(cudaMalloc(&C, (size_t)m * n * 4));
        CU(cudaDeviceSynchronize());
        const float tc = gpu_us(1, m, n, k, A, B, C), tf = gpu_us(1, m, n, k, A, Bf, C), t12 = gpu_us(0, m, n, k, A, B, C);
        printf("%5d x %-6d %10.1f %14.1f %10.1f %14.2f\n", m, k, tc, tf, t12, tf / t12);
        cudaFree(A);
        cudaFree(B);
        cudaFree(Bf);
        cudaFree(C);
        free(hA);
        free(hB);
    }

    /* 3. same-sign data */
    static const int ss[][3] = {{4096, 256, 16384}, {7344, 512, 4096}, {2048, 64, 1024}};
    for (size_t s = 0; s < sizeof ss / sizeof *ss; ++s) {
        const int m = ss[s][0], n = ss[s][1], k = ss[s][2];
        uint16_t *hA = malloc((size_t)m * k * 2), *hB = malloc((size_t)n * k * 2);
        for (size_t i = 0; i < (size_t)m * k; ++i) hA[i] = f2bf(0.5f + 0.5f * (rnd() / 4294967296.0f));
        for (size_t i = 0; i < (size_t)n * k; ++i) hB[i] = f2bf(0.5f + 0.5f * (rnd() / 4294967296.0f));
        void *A = to_device(hA, (size_t)m * k * 2), *B = to_device(hB, (size_t)n * k * 2), *C;
        CU(cudaMalloc(&C, (size_t)m * n * 4));
        CU(cudaDeviceSynchronize());
        const Counts c0 = counts(API_GEMMEX);
        gemm(1, m, n, k, A, B, C);
        CU(cudaStreamSynchronize(st));
        const Counts c1 = counts(API_GEMMEX);
        float *hC = from_device(C, (size_t)m * n * 4);
        double worst = 0, sum = 0;
        for (int e = 0; e < 256; ++e) {
            const int i = (int)(((long long)e * 7919) % m), j = (int)(((long long)e * 104729) % n);
            double x = 0, mag = 0;
            for (int l = 0; l < k; ++l) {
                const double p = (double)bf2f(hA[l + (size_t)i * k]) * (double)bf2f(hB[l + (size_t)j * k]);
                x += p;
                mag += fabs(p);
            }
            const double u = ((double)hC[i + (size_t)j * m] - x) / (mag * 0x1p-24);
            if (fabs(u) > worst) worst = fabs(u);
            sum += u;
        }
        printf("same-sign TN %d x %d x %d: largest error %.1f units, mean %.1f units (served by the tensor-core path: %s)\n",
               m, n, k, worst, sum / 256, !rt_counters ? "not counted" : delta(&c0, &c1, SERVED_TC) == 1 ? "yes" : "NO");
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
        free(hA);
        free(hB);
        free(hC);
    }
    printf("ok bf16-cost\n");
    return 0;
}
