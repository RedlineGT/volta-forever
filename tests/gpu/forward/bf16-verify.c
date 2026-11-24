/* Verify mode (VF_CUBLAS_BF16_VERIFY=1): every call the skinny path serves is also made through cuBLAS 12.9, with
 * its output in a scratch buffer, and the two compared bit for bit on the device; the tensor-core path's fallback
 * elements too. Skinny calls of each kind: cublasGemmEx TN, NN and NT, fp32 and bf16 outputs, beta 0 and beta 0.3
 * in place (C read), alpha and beta on the device, C 64 bytes past a 256-byte boundary; cublasLtMatmul (NT) with a
 * bias epilogue and a workspace, and with bf16 output, beta 0.5 and C apart from D. Then three of them captured into a
 * graph replayed twice, and a tensor-core call with failing columns. Each must be verified (the counters of a test
 * build of the runtime, vf_bf16_test_verify: one call per eager call and per replay), none may differ (the skinny
 * path gives 12.9's bits; the fallback's chains are 12.9's sequential kernel's), and the outputs the program gets
 * must be 12.9's bits (cuBLAS 12.9 called directly with the same arguments, compared on the host). A fat binary
 * whose skinny kernel changes the last bit of element (0, 0) (VF_BF16_FATBIN=control-PERTURB.fatbin) makes it fail:
 * every skinny call then differs, in that element. Prints "ok bf16-verify". */
#include "bf16-common.h"

static cublasHandle_t hv, h12;
static cublasLtHandle_t ltv, l12;
static cudaStream_t st;
static int (*rt_verify)(unsigned long long *, int);

typedef struct {
    unsigned long long v[7];
} VCounts;
static VCounts vcounts(void)
{
    VCounts c;
    memset(&c, 0, sizeof c);
    CU(cudaStreamSynchronize(st));
    if (rt_verify && rt_verify(c.v, 7) != 7) {
        printf("FAIL  verify mode's counters could not be read\n");
        exit(1);
    }
    return c;
}

/* One skinny call through cublasGemmEx (the variants), the same through 12.9 directly, compared on the host. */
static int legacy_case(const char *what, int ta, int tb, int m, int n, int k, int bf16_out, float alpha, float beta,
                       int device_scalars, size_t c_offset)
{
    const int lda = ta ? k : m, ldb = tb ? n : k, ldc = m;
    uint16_t *hA = bf16_random((size_t)m * k, 0.05), *hB = bf16_random((size_t)n * k, 1.0);
    const size_t es = bf16_out ? 2 : 4, cbytes = (size_t)ldc * n * es;
    void *hC = malloc(cbytes);
    for (size_t i = 0; i < (size_t)ldc * n; ++i) {
        if (bf16_out)
            ((uint16_t *)hC)[i] = f2bf((float)gauss());
        else
            ((float *)hC)[i] = (float)gauss();
    }
    void *dA = to_device(hA, (size_t)m * k * 2), *dB = to_device(hB, (size_t)n * k * 2);
    char *cv, *c12;
    CU(cudaMalloc((void **)&cv, cbytes + 256));
    CU(cudaMalloc((void **)&c12, cbytes + 256));
    cv += c_offset, c12 += c_offset;
    CU(cudaMemcpy(cv, hC, cbytes, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(c12, hC, cbytes, cudaMemcpyHostToDevice));
    float *ds;
    CU(cudaMalloc((void **)&ds, 2 * sizeof(float)));
    const float hs[2] = {alpha, beta};
    CU(cudaMemcpy(ds, hs, sizeof hs, cudaMemcpyHostToDevice));
    if (device_scalars) {
        cublasSetPointerMode(hv, CUBLAS_POINTER_MODE_DEVICE);
        c129.SetPointerMode(h12, CUBLAS_POINTER_MODE_DEVICE);
    }
    const void *pa = device_scalars ? (const void *)ds : (const void *)&hs[0];
    const void *pb = device_scalars ? (const void *)(ds + 1) : (const void *)&hs[1];
    const cudaDataType ct = bf16_out ? CUDA_R_16BF : CUDA_R_32F;
    const cublasStatus_t s1 = cublasGemmEx(hv, ta ? CUBLAS_OP_T : CUBLAS_OP_N, tb ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, pa,
                                           dA, CUDA_R_16BF, lda, dB, CUDA_R_16BF, ldb, pb, cv, ct, ldc,
                                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    const cublasStatus_t s2 = c129.GemmEx(h12, ta ? CUBLAS_OP_T : CUBLAS_OP_N, tb ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k,
                                          pa, dA, CUDA_R_16BF, lda, dB, CUDA_R_16BF, ldb, pb, c12, ct, ldc,
                                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CU(cudaStreamSynchronize(st));
    if (device_scalars) {
        cublasSetPointerMode(hv, CUBLAS_POINTER_MODE_HOST);
        c129.SetPointerMode(h12, CUBLAS_POINTER_MODE_HOST);
    }
    void *r1 = from_device(cv, cbytes), *r2 = from_device(c12, cbytes);
    const size_t nd = ndiff(r1, r2, cbytes, es);
    printf("case %s: TN-style %c%c %d x %d x %d, %s out, alpha %g beta %g%s, C at %zu mod 256: %zu elements differ from "
           "12.9's\n", what, ta ? 'T' : 'N', tb ? 'T' : 'N', m, n, k, bf16_out ? "bf16" : "fp32", alpha, beta,
           device_scalars ? " (device)" : "", c_offset, nd);
    CHECK(s1 == CUBLAS_STATUS_SUCCESS && s2 == CUBLAS_STATUS_SUCCESS, "%s: status %d, 12.9 %d", what, s1, s2);
    CHECK(nd == 0, "%s: %zu elements of the program's output differ from 12.9's", what, nd);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(cv - c_offset);
    cudaFree(c12 - c_offset);
    cudaFree(ds);
    free(hA);
    free(hB);
    free(hC);
    free(r1);
    free(r2);
    return 1;
}

/* One skinny call through cublasLtMatmul (NT: A m x k, B n x k transposed), the same through 12.9 directly, compared
 * on the host. */
static int lt_case(const char *what, int m, int n, int k, int bf16_out, int bias, float beta, int c_apart)
{
    const cudaDataType ct = bf16_out ? CUDA_R_16BF : CUDA_R_32F;
    const size_t es = bf16_out ? 2 : 4, cbytes = (size_t)m * n * es;
    uint16_t *hA = bf16_random((size_t)m * k, 0.05), *hB = bf16_random((size_t)n * k, 1.0);
    void *hC = malloc(cbytes), *hbias = malloc((size_t)m * es);
    for (size_t i = 0; i < (size_t)m * n; ++i) {
        if (bf16_out)
            ((uint16_t *)hC)[i] = f2bf((float)gauss());
        else
            ((float *)hC)[i] = (float)gauss();
    }
    for (int i = 0; i < m; ++i) {
        if (bf16_out)
            ((uint16_t *)hbias)[i] = f2bf((float)gauss());
        else
            ((float *)hbias)[i] = (float)gauss();
    }
    void *dA = to_device(hA, (size_t)m * k * 2), *dB = to_device(hB, (size_t)n * k * 2), *dbias = to_device(hbias, (size_t)m * es);
    void *Cv = to_device(hC, cbytes), *C12 = to_device(hC, cbytes), *Dv = Cv, *D12 = C12, *ws1, *ws2;
    if (c_apart) {
        CU(cudaMalloc(&Dv, cbytes));
        CU(cudaMalloc(&D12, cbytes));
    }
    CU(cudaMalloc(&ws1, 4u << 20));
    CU(cudaMalloc(&ws2, 4u << 20));
    cublasLtMatmulDesc_t d;
    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const int32_t tB = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof tB);
    if (bias) {
        const uint32_t ep = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof ep);
        cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dbias, sizeof dbias);
    }
    cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, m, k, m);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, n, k, n);
    cublasLtMatrixLayoutCreate(&lc, ct, m, n, m);
    const float one = 1.0f;
    const cublasStatus_t s1 = cublasLtMatmul(ltv, d, &one, dA, la, dB, lb, &beta, Cv, lc, Dv, lc, NULL, ws1, 4u << 20, st);
    const cublasStatus_t s2 = c129.LtMatmul(l12, d, &one, dA, la, dB, lb, &beta, C12, lc, D12, lc, NULL, ws2, 4u << 20, st);
    CU(cudaStreamSynchronize(st));
    void *r1 = from_device(Dv, cbytes), *r2 = from_device(D12, cbytes);
    const size_t nd = ndiff(r1, r2, cbytes, es);
    printf("case %s: cublasLtMatmul NT %d x %d x %d, %s out%s, beta %g%s: %zu elements differ from 12.9's\n", what, m, n, k,
           bf16_out ? "bf16" : "fp32", bias ? ", bias" : "", beta, c_apart ? ", C apart from D" : "", nd);
    CHECK(s1 == CUBLAS_STATUS_SUCCESS && s2 == CUBLAS_STATUS_SUCCESS, "%s: status %d, 12.9 %d", what, s1, s2);
    CHECK(nd == 0, "%s: %zu elements of the program's output differ from 12.9's", what, nd);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(d);
    if (c_apart) {
        cudaFree(Dv);
        cudaFree(D12);
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dbias);
    cudaFree(Cv);
    cudaFree(C12);
    cudaFree(ws1);
    cudaFree(ws2);
    free(hA);
    free(hB);
    free(hC);
    free(hbias);
    free(r1);
    free(r2);
    return 1;
}

int main(void)
{
    CU(cudaSetDevice(0));
    CU(cudaFree(0));
    open_129();
    CU(cudaStreamCreate(&st)); /* blocking: ordered after the uploads */
    if (cublasCreate(&hv) || c129.Create(&h12) || cublasLtCreate(&ltv) || c129.LtCreate(&l12)) return 1;
    open_runtime();
    void *rt = dlopen("libvf_cublas_bf16.so", RTLD_NOW | RTLD_NOLOAD);
    if (rt) *(void **)&rt_verify = dlsym(rt, "vf_bf16_test_verify");
    if (!rt_verify) printf("note  a release runtime: verify mode's counters are not checked\n");
    cublasSetStream(hv, st);
    c129.SetStream(h12, st);
    const VCounts v0 = vcounts();
    int skinny = 0;
    skinny += legacy_case("TN fp32", 1, 0, 1536, 8, 1536, 0, 1.0f, 0.0f, 0, 0);
    skinny += legacy_case("TN fp32, C read", 1, 0, 1536, 8, 1536, 0, 0.7f, 0.3f, 0, 0);
    skinny += legacy_case("TN bf16", 1, 0, 1536, 16, 1536, 1, 1.0f, 0.0f, 0, 0);
    skinny += legacy_case("NN fp32", 0, 0, 1536, 4, 1920, 0, 1.0f, 0.0f, 0, 0);
    skinny += legacy_case("NT bf16, C read", 0, 1, 1536, 8, 1920, 1, 1.25f, 0.5f, 0, 0);
    skinny += legacy_case("TN fp32, device scalars", 1, 0, 1536, 8, 1536, 0, 0.7f, 0.5f, 1, 0);
    skinny += legacy_case("TN fp32, C at 64", 1, 0, 1536, 8, 1536, 0, 1.0f, 0.25f, 0, 64);
    skinny += lt_case("Lt bias", 1536, 8, 1920, 0, 1, 0.0f, 0);
    skinny += lt_case("Lt bf16, C apart", 1536, 12, 1920, 1, 0, 0.5f, 1);
    const VCounts v1 = vcounts();
    /* three of them captured into a graph, replayed twice */
    const int M = 1536, N = 8, K = 1920;
    uint16_t *hA = bf16_random((size_t)M * K, 0.05), *hB = bf16_random((size_t)N * K, 1.0);
    void *dA = to_device(hA, (size_t)M * K * 2), *dB = to_device(hB, (size_t)N * K * 2), *dC, *dD, *ws;
    CU(cudaMalloc(&dC, (size_t)M * N * 4));
    CU(cudaMalloc(&dD, (size_t)M * N * 4));
    CU(cudaMalloc(&ws, 4u << 20));
    CU(cudaMemset(dC, 0, (size_t)M * N * 4));
    cublasLtMatmulDesc_t d;
    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const int32_t tB = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof tB);
    cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, M, K, M);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, N, K, N);
    cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, M, N, M);
    const float one = 1.0f, zero = 0.0f, half = 0.5f;
    cudaGraph_t g;
    cudaGraphExec_t e;
    CU(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    cublasGemmEx(hv, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, &one, dA, CUDA_R_16BF, M, dB, CUDA_R_16BF, N, &zero, dC, CUDA_R_32F,
                 M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cublasGemmEx(hv, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, &one, dA, CUDA_R_16BF, M, dB, CUDA_R_16BF, N, &half, dC, CUDA_R_32F,
                 M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cublasLtMatmul(ltv, d, &one, dA, la, dB, lb, &half, dC, lc, dD, lc, NULL, ws, 4u << 20, st);
    CU(cudaStreamEndCapture(st, &g));
    CU(cudaGraphInstantiate(&e, g, 0));
    CU(cudaGraphLaunch(e, st));
    CU(cudaGraphLaunch(e, st));
    const VCounts v2 = vcounts();
    /* a tensor-core call with 12 failing columns: its fallback elements verified */
    const int TM = 4096, TN_ = 128, TK = 4096;
    uint16_t *tA16 = bf16_random((size_t)TM * TK, 0.05), *tB16 = bf16_random((size_t)TN_ * TK, 1.0);
    for (int c = 0; c < 12; ++c) tB16[(size_t)(c * 7) + (size_t)(c * 10) * TK] = f2bf(1e-30f);
    void *tcA = to_device(tA16, (size_t)TM * TK * 2), *tcB = to_device(tB16, (size_t)TN_ * TK * 2), *tC;
    CU(cudaMalloc(&tC, (size_t)TM * TN_ * 4));
    cublasGemmEx(hv, CUBLAS_OP_T, CUBLAS_OP_N, TM, TN_, TK, &one, tcA, CUDA_R_16BF, TK, tcB, CUDA_R_16BF, TK, &zero, tC,
                 CUDA_R_32F, TM, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    const VCounts v3 = vcounts();
    printf("verify mode: eager skinny calls %llu verified, %llu differed (%llu elements); captured: %llu verified in two "
           "replays, %llu differed; tensor-core call: %llu verified, %llu differed; %llu signatures, %llu not verified\n",
           v1.v[0] - v0.v[0], v1.v[1] - v0.v[1], v1.v[2] - v0.v[2], v2.v[0] - v1.v[0], v2.v[1] - v1.v[1],
           v3.v[0] - v2.v[0], v3.v[1] - v2.v[1], v3.v[5], v3.v[4]);
    if (rt_verify) {
        CHECK(v1.v[0] - v0.v[0] == (unsigned long long)skinny, "eager skinny calls: %llu verified, not %d",
              v1.v[0] - v0.v[0], skinny);
        CHECK(v2.v[0] - v1.v[0] == 6, "captured skinny calls: %llu verified in two replays of three, not 6",
              v2.v[0] - v1.v[0]);
        CHECK(v3.v[0] - v2.v[0] == 1, "the tensor-core call: %llu verified, not 1", v3.v[0] - v2.v[0]);
        CHECK(v3.v[1] == 0 && v3.v[2] == 0, "%llu calls differed from 12.9's, in %llu elements", v3.v[1], v3.v[2]);
        CHECK(v3.v[4] == 0, "%llu calls not verified", v3.v[4]);
    }
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(d);
    if (failures) {
        printf("%d checks failed\n", failures);
        return 1;
    }
    printf("ok bf16-verify\n");
    return 0;
}
