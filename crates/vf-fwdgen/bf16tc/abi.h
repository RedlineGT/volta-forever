/* Kernel arguments of the bf16 cuBLAS runtime (crates/vf-fwdgen/bf16tc/runtime.c, gcc) and its kernels
 * (crates/vf-fwdgen/bf16tc/kernels.cu, nvcc). Plain C types only, so that both compilers lay them out alike
 * (x86-64 LP64): the runtime passes each structure by value as the kernel's only parameter. */
#ifndef VF_BF16_ABI_H
#define VF_BF16_ABI_H

#define VF_BF16_SENT (-32768)              /* exponent of a row or column that failed the check */
#define VF_BF16_FB_TN_SMEM (44 * 1024)     /* dynamic shared memory of vf_bf16_fallback_tn */

/* One operand of the tensor-core path: R vectors of length k along the reduction dimension.
 * Element l of vector v is src[v * vstride + l * lstride] (bf16 bits). Its fp16 copy is
 * dst[v * dvstride + l * dlstride] for l < kp (zero for k <= l < kp). */
typedef struct {
    const unsigned short *src;
    long long vstride, lstride;
    unsigned short *dst;
    long long dvstride, dlstride;
    int *e;          /* exponent per vector, or VF_BF16_SENT */
    int *list;       /* indices (base + v) of the failed vectors, in no particular order */
    int *count;      /* number of failed vectors (zeroed by the runtime once per call); NULL: not counted */
    int R;           /* number of vectors (0: segment absent) */
    int contig;      /* lstride == 1: one warp per vector */
    int vec;         /* contig and 16-byte loads possible */
    int nblocks;     /* blocks of the launch that serve this segment */
    int base;        /* index in the call of vector 0 (the chunk's first row or column) */
    int lim;         /* entries of list (the call's rows or columns) */
} VfSeg;

typedef struct {
    VfSeg seg[2];    /* 0: rows of op(A), 1: columns of op(B) */
    int k, kp;
} VfScaleArgs;

/* Output side, shared by the epilogue, the fallback and the skinny kernels.
 * D(i, j) = alpha * acc, then + beta * C(i, j) by one fused multiply-add when beta != 0, then
 * + bias[i]. C is read only when beta != 0. */
typedef struct {
    const void *C;
    long long ldc;
    void *D;
    long long ldd;
    const void *bias;
    const float *alpha_p, *beta_p;  /* device pointer mode: read when the kernel runs */
    float alpha, beta;              /* host pointer mode */
    int dtype;                      /* C and D: 0 fp32, 1 bf16 */
    int btype;                      /* bias: -1 none, 0 fp32, 1 bf16 */
    int beta_mode;                  /* 0: host beta == 0, 1: host beta != 0, 2: device */
    int pad_;
} VfOut;

typedef struct {
    const float *cp;      /* partials: element (i, j) of slice s at cp[s * cpstride + j * mc + i] */
    long long cpstride;
    const int *eA, *eB;
    int ns, mc, n, pad_;
    VfOut o;
} VfEpiArgs;

typedef struct {
    const unsigned short *A;   /* op(A)(i, l) = A[i * a_si + l * a_sl] */
    long long a_si, a_sl;
    const unsigned short *B;   /* op(B)(l, j) = B[l * b_sl + j * b_sj] */
    long long b_sl, b_sj;
    const int *cntA, *listA, *eA;
    const int *cntB, *listB, *eB;
    int mc, n, k, pad_;
    VfOut o;
} VfFbArgs;

/* The skinny path: every element is one fp32 fused multiply-add chain over l = 0..k-1 of the bf16
 * operands, as in the fallback; no workspace. */
typedef struct {
    const unsigned short *A;   /* op(A)(i, l) = A[i * a_si + l * a_sl] */
    long long a_si, a_sl;
    const unsigned short *B;   /* op(B)(l, j) = B[l * b_sl + j * b_sj] */
    long long b_sl, b_sj;
    int m, n, k, pad_;
    VfOut o;
} VfSkArgs;

#endif
