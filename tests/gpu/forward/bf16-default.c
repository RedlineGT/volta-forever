/* What a program the variants do not serve pays for them, and what a program that forks sees (measurement
 * for the question of a default; run by hand on a V100, once with the variants first in the library path
 * and the switch on, once with the default forwarders):
 *   bf16-default overhead   host time per call of tiny calls that are forwarded: cublasGemmEx with fp32
 *                           operands (8 x 8 x 8), cublasGemmEx with bf16 operands and one column, and
 *                           cublasSgemm (not intercepted); the same calls given to cuBLAS 12.9 directly;
 *                           mean over 200,000 calls, ns per call
 *   bf16-default setup      time and device memory of the first cublasCreate, and of the first call the
 *                           skinny path serves (TN 1536 x 8 x 1536)
 *   bf16-default fork       a served call, then fork: a child that exits at once, a child that makes a
 *                           call through the variants, a child that makes it through 12.9 directly
 *                           (each with a 20 s alarm); the parent's served calls after them, with their bits */
#include "bf16-common.h"
#include <signal.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static double now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static __typeof__(cublasSgemm_v2) *sgemm129;

int main(int argc, char **argv)
{
    if (argc < 2) return 2;
    const char *mode = argv[1];
    if (!strcmp(mode, "setup")) {
        size_t f0 = 0, f_create = 0, f_alloc = 0, f_call = 0, total = 0;
        CU(cudaSetDevice(0));
        CU(cudaFree(0));
        cudaDeviceSynchronize();
        cudaMemGetInfo(&f0, &total);
        cublasHandle_t h;
        const double t0 = now_ns();
        const int st = (int)cublasCreate(&h);
        const double t1 = now_ns();
        cudaDeviceSynchronize();
        cudaMemGetInfo(&f_create, &total);
        uint16_t *hA = bf16_random(1536 * 1536, 0.05), *hB = bf16_random(8 * 1536, 1.0);
        void *A = to_device(hA, 1536 * 1536 * 2), *B = to_device(hB, 8 * 1536 * 2), *C;
        CU(cudaMalloc(&C, 1536 * 8 * 4));
        cudaDeviceSynchronize();
        cudaMemGetInfo(&f_alloc, &total);
        const float one = 1.0f, zero = 0.0f;
        const double t2 = now_ns();
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 1536, 8, 1536, &one, A, CUDA_R_16BF, 1536, B, CUDA_R_16BF, 1536, &zero, C,
                     CUDA_R_32F, 1536, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        const double t3 = now_ns();
        cudaDeviceSynchronize();
        cudaMemGetInfo(&f_call, &total);
        printf("setup: first cublasCreate %.2f ms (status %d), device memory taken %lld bytes; first skinny-sized call "
               "%.2f ms of host time, device memory taken %lld bytes\n",
               (t1 - t0) / 1e6, st, (long long)f0 - (long long)f_create, (t3 - t2) / 1e6,
               (long long)f_alloc - (long long)f_call);
        return 0;
    }
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    void *b12 = dlopen("libcublas.so.12", RTLD_NOW | RTLD_NOLOAD);
    if (b12) *(void **)&sgemm129 = dlsym(b12, "cublasSgemm_v2");
    cublasHandle_t hv, h12;
    if (cublasCreate(&hv) || c129.Create(&h12)) return 1;
    open_runtime();
    if (!strcmp(mode, "overhead")) {
        void *a, *b, *c;
        CU(cudaMalloc(&a, 1 << 16));
        CU(cudaMalloc(&b, 1 << 16));
        CU(cudaMalloc(&c, 1 << 16));
        const float one = 1.0f, zero = 0.0f;
        const int N = 200000;
        static const char *const names[] = {"cublasGemmEx fp32 8x8x8", "cublasGemmEx bf16 n = 1", "cublasSgemm 8x8x8"};
        for (int kind = 0; kind < 3; ++kind)
            for (int direct = 0; direct < 2; ++direct) {
                cublasHandle_t h = direct ? h12 : hv;
                double total = 0;
                for (int i = -1000; i < N; ++i) {
                    const double t0 = now_ns();
                    if (kind == 0) {
                        if (direct)
                            c129.GemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, 8, 8, 8, &one, a, CUDA_R_32F, 8, b, CUDA_R_32F, 8,
                                        &zero, c, CUDA_R_32F, 8, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
                        else
                            cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, 8, 8, 8, &one, a, CUDA_R_32F, 8, b, CUDA_R_32F, 8,
                                         &zero, c, CUDA_R_32F, 8, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
                    } else if (kind == 1) {
                        if (direct)
                            c129.GemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 64, 1, 64, &one, a, CUDA_R_16BF, 64, b, CUDA_R_16BF,
                                        64, &zero, c, CUDA_R_32F, 64, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
                        else
                            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 64, 1, 64, &one, a, CUDA_R_16BF, 64, b, CUDA_R_16BF,
                                         64, &zero, c, CUDA_R_32F, 64, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
                    } else {
                        if (direct && sgemm129)
                            sgemm129(h, CUBLAS_OP_N, CUBLAS_OP_N, 8, 8, 8, &one, a, 8, b, 8, &zero, c, 8);
                        else if (!direct)
                            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 8, 8, 8, &one, a, 8, b, 8, &zero, c, 8);
                    }
                    const double t1 = now_ns();
                    if (i >= 0) total += t1 - t0;
                    if ((i & 1023) == 1023) cudaDeviceSynchronize();
                }
                cudaDeviceSynchronize();
                printf("overhead: %-26s %-12s %7.0f ns per call\n", names[kind], direct ? "12.9 directly" : "through VF",
                       total / N);
            }
        return 0;
    }
    if (!strcmp(mode, "fork")) {
        const int M = 1536, NS = 8, K = 1536;
        uint16_t *hA = bf16_random((size_t)M * K, 0.05), *hB = bf16_random((size_t)NS * K, 1.0);
        void *A = to_device(hA, (size_t)M * K * 2), *B = to_device(hB, (size_t)NS * K * 2), *C, *C12;
        CU(cudaMalloc(&C, (size_t)M * NS * 4));
        CU(cudaMalloc(&C12, (size_t)M * NS * 4));
        const float one = 1.0f, zero = 0.0f;
        cublasGemmEx(hv, CUBLAS_OP_T, CUBLAS_OP_N, M, NS, K, &one, A, CUDA_R_16BF, K, B, CUDA_R_16BF, K, &zero, C,
                     CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        c129.GemmEx(h12, CUBLAS_OP_T, CUBLAS_OP_N, M, NS, K, &one, A, CUDA_R_16BF, K, B, CUDA_R_16BF, K, &zero, C12,
                    CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        fflush(stdout);
        static const char *const what[] = {"exits at once", "calls through the variants", "calls cuBLAS 12.9 directly"};
        for (int child = 0; child < 3; ++child) {
            const pid_t p = fork();
            if (p == 0) {
                alarm(20);
                if (child == 0) _exit(0);
                const cublasStatus_t s =
                    child == 1 ? cublasGemmEx(hv, CUBLAS_OP_T, CUBLAS_OP_N, M, NS, K, &one, A, CUDA_R_16BF, K, B, CUDA_R_16BF,
                                              K, &zero, C, CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
                               : c129.GemmEx(h12, CUBLAS_OP_T, CUBLAS_OP_N, M, NS, K, &one, A, CUDA_R_16BF, K, B, CUDA_R_16BF,
                                             K, &zero, C12, CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
                const cudaError_t e = cudaDeviceSynchronize();
                printf("fork: the child that %s: status %d, then %s\n", what[child], (int)s, cudaGetErrorName(e));
                fflush(stdout);
                _exit(0);
            }
            int ws = 0;
            waitpid(p, &ws, 0);
            printf("fork: the child that %s ended: %s %d\n", what[child],
                   WIFEXITED(ws) ? "exit status" : "signal", WIFEXITED(ws) ? WEXITSTATUS(ws) : WTERMSIG(ws));
        }
        const Counts k0 = counts(API_GEMMEX);
        cudaMemset(C, 0, (size_t)M * NS * 4);
        cublasGemmEx(hv, CUBLAS_OP_T, CUBLAS_OP_N, M, NS, K, &one, A, CUDA_R_16BF, K, B, CUDA_R_16BF, K, &zero, C,
                     CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        const cudaError_t e = cudaDeviceSynchronize();
        const Counts k1 = counts(API_GEMMEX);
        float *r = from_device(C, (size_t)M * NS * 4), *r12 = from_device(C12, (size_t)M * NS * 4);
        printf("fork: the parent afterwards: %s, served by the skinny path %s, %s 12.9's bits\n", cudaGetErrorName(e),
               !rt_counters ? "(not counted)" : delta(&k0, &k1, SERVED_SKINNY) == 1 ? "yes" : "NO",
               memcmp(r, r12, (size_t)M * NS * 4) ? "NOT" : "with");
        return 0;
    }
    return 2;
}
