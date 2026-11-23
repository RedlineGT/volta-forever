// Kernels of the bf16 cuBLAS runtime (crates/vf-fwdgen/bf16tc/runtime.c): exact power-of-two scaling of bf16
// operands to fp16 under the strict check, the epilogue over fixed k-slice partials, the fallback
// for the rows and columns that fail the check, and the skinny path. docs/emulations.md, "cuBLAS
// bf16 paths", states the arithmetic.
//
// Built by scripts/build-bf16-kernels.sh into kernels.fatbin (sm_70 machine code only), which is
// checked in with kernels.manifest. The source uses no routine of NVIDIA's headers: conversions are
// written here, and every rounding operation is an explicitly rounded one (__fmaf_rn, __fmul_rn,
// __fadd_rn, cvt.rn), so that no rounding is the compiler's choice (the build adds -fmad=false).
// What nvcc's forced cuda_runtime.h provides and is used here is one instruction each: __ldg,
// __shfl_xor_sync, __any_sync, atomicAdd (to append to the lists of failed vectors, whose order
// does not matter), __uint_as_float, __float_as_uint. No module variable, no counter.
#include "abi.h"

// Controls (tests/gpu/forward-bf16.sh builds them apart, never into kernels.fatbin): each breaks one
// property so that the test of that property can be seen to fail.
#if defined(VF_KERNEL_CONTROL_MULADD)  // the chains as separate multiply and add
#define CHAIN(a, b, acc) __fadd_rn(__fmul_rn((a), (b)), (acc))
#else  // one fused multiply-add per term: the product is exact, one rounding per step
#define CHAIN(a, b, acc) __fmaf_rn((a), (b), (acc))
#endif

static __device__ __forceinline__ unsigned umax32(unsigned a, unsigned b) { return a > b ? a : b; }
static __device__ __forceinline__ int imin32(int a, int b) { return a < b ? a : b; }
static __device__ __forceinline__ int imax32(int a, int b) { return a > b ? a : b; }
static __device__ __forceinline__ int cdiv(int a, int b) { return (a + b - 1) / b; }

static __device__ __forceinline__ unsigned warp_max_u(unsigned x)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x = umax32(x, __shfl_xor_sync(0xffffffffu, x, o));
    return x;
}

// bf16 to fp32: exact (the fp32 value with the same 16 high bits).
static __device__ __forceinline__ float bf2f(unsigned b) { return __uint_as_float(b << 16); }

// fp32 to bf16, rounded to nearest even; every NaN becomes 0x7fff (what cuBLAS 12.9 writes on
// this GPU; tests/gpu/forward/bf16-arith.c).
static __device__ __forceinline__ unsigned short f2bf(float v)
{
    const unsigned u = __float_as_uint(v);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fffu;
    return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// fp32 to fp16 rounded to nearest even, and fp16 to fp32 (exact): the hardware conversions.
static __device__ __forceinline__ unsigned short f2h_rn(float y)
{
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(y));
    return h;
}
static __device__ __forceinline__ float h2f(unsigned short h)
{
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

// 2^s as a normal fp32 (s in [-126, 127]).
static __device__ __forceinline__ float pow2f(int s) { return __uint_as_float((unsigned)(s + 127) << 23); }

static __device__ __forceinline__ uint4 zero4()
{
    uint4 z;
    z.x = z.y = z.z = z.w = 0u;
    return z;
}

// Strict check of one element: returns 1 unless the bf16 value (bits b) times the power of two
// `scale` is exactly an fp16. Within the exponent window the product is exact in fp32 (no fp32
// underflow), so comparing the fp16 round trip with it is exact; a nonzero value must not
// become zero.
static __device__ __forceinline__ unsigned conv1(unsigned b, float scale, unsigned short &h)
{
    const float y = __fmul_rn(bf2f(b), scale);
    h = f2h_rn(y);
    const float back = h2f(h);
    return (unsigned)(back != y) | (unsigned)(((b & 0x7fffu) != 0u) & (back == 0.0f));
}

static __device__ __forceinline__ unsigned conv2(unsigned w, float scale, unsigned &out)
{
    unsigned short h0, h1;
    const unsigned bad = conv1(w & 0xffffu, scale, h0) | conv1(w >> 16, scale, h1);
    out = (unsigned)h0 | ((unsigned)h1 << 16);
    return bad;
}

static __device__ __forceinline__ unsigned absmax2(unsigned w) { return umax32(w & 0x7fffu, (w >> 16) & 0x7fffu); }

// From the largest |bits| of a vector: 1 if it fails before conversion (inf/NaN present, maximum
// subnormal, or exponent outside the window [-24, 31]); else e = exponent of the maximum
// (15 for an all-zero vector, which converts exactly with scale 1).
static __device__ __forceinline__ int classify(unsigned mx, int &e)
{
    e = 15;
    if (mx >= 0x7f80u) return 1;   // an infinity or a NaN
    if (mx == 0u) return 0;        // all elements are zeros
    const int bexp = (int)(mx >> 7);
    if (bexp == 0) return 1;       // subnormal maximum: exponent below -126
    e = bexp - 127;
    return (e < -24) | (e > 31);
}

static __device__ __forceinline__ void finish(const VfSeg &s, int v, int bad, int e)
{
    s.e[v] = bad ? VF_BF16_SENT : e;
    if (bad) {
        const int pos = atomicAdd(s.count, 1);
        if (pos >= 0 && pos < s.R) s.list[pos] = v;
    }
}

// Vectors contiguous along k: one warp per vector, two passes (maximum, then convert + check).
static __device__ __forceinline__ void scale_contig(const VfSeg &s, int blk, const VfScaleArgs &a)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int v = blk * 8 + warp;
    if (v >= s.R) return;
    const unsigned short *p = s.src + (long long)v * s.vstride;
    unsigned short *q = s.dst + (long long)v * s.dvstride;
    const int k = a.k, kp = a.kp;
    const int k8 = s.vec ? (k & ~7) : 0;
    unsigned mx = 0;
    if (s.vec) {
        const uint4 *p4 = reinterpret_cast<const uint4 *>(p);
        for (int i = lane; i < (k8 >> 3); i += 32) {
            const uint4 w = __ldg(p4 + i);
            mx = umax32(mx, umax32(umax32(absmax2(w.x), absmax2(w.y)), umax32(absmax2(w.z), absmax2(w.w))));
        }
    }
    for (int l = k8 + lane; l < k; l += 32) mx = umax32(mx, (unsigned)__ldg(p + l) & 0x7fffu);
    mx = warp_max_u(mx);
    int e;
    int bad = classify(mx, e);
    if (bad) {
        for (int l = lane; l < kp; l += 32) q[l] = 0;
    } else {
        const float scale = pow2f(15 - e);
        unsigned badl = 0;
        if (s.vec) {
            const uint4 *p4 = reinterpret_cast<const uint4 *>(p);
            uint4 *q4 = reinterpret_cast<uint4 *>(q);
            for (int i = lane; i < (k8 >> 3); i += 32) {
                const uint4 w = __ldg(p4 + i);
                uint4 o;
                badl |= conv2(w.x, scale, o.x) | conv2(w.y, scale, o.y) | conv2(w.z, scale, o.z) |
                        conv2(w.w, scale, o.w);
                q4[i] = o;
            }
        }
        for (int l = k8 + lane; l < k; l += 32) {
            unsigned short h;
            badl |= conv1(__ldg(p + l), scale, h);
            q[l] = h;
        }
        for (int l = k + lane; l < kp; l += 32) q[l] = 0;
        bad = __any_sync(0xffffffffu, badl != 0u);
    }
    if (lane == 0) finish(s, v, bad, e);
}

// Vectors strided along k (unit stride between vectors): a block takes 64 vectors, its 8 warps
// split k; maxima and failures are reduced through shared memory.
static __device__ __forceinline__ void scale_strided(const VfSeg &s, int blk, const VfScaleArgs &a)
{
    __shared__ unsigned sred[8][64];
    __shared__ float sscale[64];
    __shared__ int sexp[64];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int v0 = blk * 64;
    const int va = v0 + lane, vb = v0 + lane + 32;
    const bool oka = va < s.R, okb = vb < s.R;
    const long long oa = (long long)va * s.vstride, ob = (long long)vb * s.vstride;
    unsigned mxa = 0, mxb = 0;
    for (int l = warp; l < a.k; l += 8) {
        const unsigned short *row = s.src + (long long)l * s.lstride;
        if (oka) mxa = umax32(mxa, (unsigned)__ldg(row + oa) & 0x7fffu);
        if (okb) mxb = umax32(mxb, (unsigned)__ldg(row + ob) & 0x7fffu);
    }
    sred[warp][lane] = mxa;
    sred[warp][lane + 32] = mxb;
    __syncthreads();
    if (threadIdx.x < 64) {
        unsigned mx = 0;
        for (int w = 0; w < 8; ++w) mx = umax32(mx, sred[w][threadIdx.x]);
        int e;
        const int bad = classify(mx, e);
        sexp[threadIdx.x] = bad ? VF_BF16_SENT : e;
        sscale[threadIdx.x] = pow2f(15 - e);
    }
    __syncthreads();
    const float sca = sscale[lane], scb = sscale[lane + 32];
    const bool cva = oka && sexp[lane] != VF_BF16_SENT, cvb = okb && sexp[lane + 32] != VF_BF16_SENT;
    unsigned bada = 0, badb = 0;
    const long long da = (long long)va * s.dvstride, db = (long long)vb * s.dvstride;
    for (int l = warp; l < a.kp; l += 8) {
        unsigned short *drow = s.dst + (long long)l * s.dlstride;
        unsigned short ha = 0, hb = 0;
        if (l < a.k) {
            const unsigned short *row = s.src + (long long)l * s.lstride;
            if (cva) bada |= conv1(__ldg(row + oa), sca, ha);
            if (cvb) badb |= conv1(__ldg(row + ob), scb, hb);
        }
        if (oka) drow[da] = ha;
        if (okb) drow[db] = hb;
    }
    __syncthreads();
    sred[warp][lane] = bada;
    sred[warp][lane + 32] = badb;
    __syncthreads();
    if (threadIdx.x < 64 && v0 + (int)threadIdx.x < s.R) {
        unsigned b = 0;
        for (int w = 0; w < 8; ++w) b |= sred[w][threadIdx.x];
        const int e = sexp[threadIdx.x];
        const int bad = (e == VF_BF16_SENT) | (b != 0u);
        finish(s, v0 + threadIdx.x, bad, e);
    }
}

extern "C" __global__ void __launch_bounds__(256) vf_bf16_scale(VfScaleArgs a)
{
    int blk = blockIdx.x;
    if (blk < a.seg[0].nblocks) {
        if (a.seg[0].contig)
            scale_contig(a.seg[0], blk, a);
        else
            scale_strided(a.seg[0], blk, a);
    } else {
        blk -= a.seg[0].nblocks;
        if (a.seg[1].contig)
            scale_contig(a.seg[1], blk, a);
        else
            scale_strided(a.seg[1], blk, a);
    }
}

// ---- output side ----------------------------------------------------------------------------

struct Scal {
    float alpha, beta;
    bool readc;
};

static __device__ __forceinline__ Scal scalars(const VfOut &o)
{
    Scal s;
    s.alpha = o.alpha_p ? *o.alpha_p : o.alpha;
    if (o.beta_mode == 2) {
        s.beta = *o.beta_p;
        s.readc = s.beta != 0.0f;
    } else {
        s.beta = o.beta;
        s.readc = o.beta_mode == 1;
    }
    return s;
}

static __device__ __forceinline__ float rd_c(const VfOut &o, long long idx)
{
    return o.dtype ? bf2f(reinterpret_cast<const unsigned short *>(o.C)[idx]) : reinterpret_cast<const float *>(o.C)[idx];
}

static __device__ __forceinline__ void wr_d(const VfOut &o, long long idx, float v)
{
    if (o.dtype)
        reinterpret_cast<unsigned short *>(o.D)[idx] = f2bf(v);
    else
        reinterpret_cast<float *>(o.D)[idx] = v;
}

static __device__ __forceinline__ float rd_bias(const VfOut &o, int i)
{
    return o.btype == 1 ? bf2f(__ldg(reinterpret_cast<const unsigned short *>(o.bias) + i))
                        : __ldg(reinterpret_cast<const float *>(o.bias) + i);
}

// v = alpha * acc already rounded; adds beta * C by one fused multiply-add, then the bias. When beta
// is zero (+0 or -0) C is not read, and +0 is added instead, as cuBLAS 12.9 does: alpha * acc = -0
// then becomes +0 (tests/gpu/forward/bf16-arith.c).
static __device__ __forceinline__ float finalize(const VfOut &o, const Scal &sc, float v, int i, long long cidx)
{
#ifdef VF_KERNEL_CONTROL_EPI_SEPARATE  // beta * C rounded before it is added
    if (sc.readc) v = __fadd_rn(v, __fmul_rn(sc.beta, rd_c(o, cidx)));
#else
    if (sc.readc) v = __fmaf_rn(sc.beta, rd_c(o, cidx), v);
#endif
#ifndef VF_KERNEL_CONTROL_NO_ZERO_TERM  // the control leaves alpha * acc as it is when beta is zero
    else v = __fadd_rn(v, 0.0f);
#endif
    if (o.btype >= 0) v = __fadd_rn(v, rd_bias(o, i));
    return v;
}

#define VF_EPI_J 8

// Tensor-core elements: partials summed in slice order 0, 1, ... (fp32), exact rescale by
// 2^(e_i + f_j - 30), then alpha, beta * C, bias. Elements of failed rows/columns are left to
// the fallback (one writer per element; an in-place C stays intact for it).
extern "C" __global__ void __launch_bounds__(128, 8) vf_bf16_epilogue(VfEpiArgs p)
{
    const int i = blockIdx.x * 128 + threadIdx.x;
    if (i >= p.mc) return;
    const int ea = p.eA[i];
#ifndef VF_KERNEL_CONTROL_EPI_ALL  // the control writes the elements of failed rows and columns too
    if (ea == VF_BF16_SENT) return;
#endif
    const Scal sc = scalars(p.o);
    const int j0 = blockIdx.y * VF_EPI_J;
    const int j1 = imin32(j0 + VF_EPI_J, p.n);
    for (int j = j0; j < j1; ++j) {
        const int eb = p.eB[j];
#ifndef VF_KERNEL_CONTROL_EPI_ALL
        if (eb == VF_BF16_SENT) continue;
#endif
        const long long off = (long long)j * p.mc + i;
        float acc = 0.0f;  // +0 start: an all-zero sum is +0, as in a plain fp32 accumulation
        for (int s = 0; s < p.ns; ++s) acc = __fadd_rn(acc, p.cp[s * p.cpstride + off]);
        float v = __fmul_rn(sc.alpha, __fmul_rn(acc, pow2f(ea + eb - 30)));
        v = finalize(p.o, sc, v, i, (long long)j * p.o.ldc + i);
        wr_d(p.o, (long long)j * p.o.ldd + i, v);
    }
}

// Fallback: tiles over (all rows of the chunk x failed columns of op(B)) and (failed rows of
// op(A) x columns that passed). Each element is one fp32 FMA chain over l = 0, 1, ..., k-1 of the
// original bf16 operands (exact products, fp32 sums): its value does not depend on the tile
// shape, the list order or the position in the tile. A persistent grid walks the tiles; with
// empty lists every block exits after reading the two counts. Counts are clamped and list
// entries bounds-checked.
template <int BM, int BN, int BK, int TM, int TN>
static __device__ __forceinline__ void fallback_body(const VfFbArgs &p)
{
    constexpr int TC = BN / TN;
    constexpr int TR = BM / TM;
    static_assert(TR * TC == 256, "256 threads");
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN + 1];
    __shared__ int rowIdx[BM];
    __shared__ int colIdx[BN];
    __shared__ unsigned char colOk[BN];
    const int tid = threadIdx.x;
    const int tr = tid / TC, tc = tid % TC;
    const int ca = imin32(imax32(*p.cntA, 0), p.mc);
    const int cb = imin32(imax32(*p.cntB, 0), p.n);
    const int mt = (p.mc + BM - 1) / BM, nt = (p.n + BN - 1) / BN;
    const int colTiles = ((cb + BN - 1) / BN) * mt;
    const int total = colTiles + ((ca + BM - 1) / BM) * nt;
    if ((int)blockIdx.x >= total) return;
    const Scal sc = scalars(p.o);
    for (int t = blockIdx.x; t < total; t += gridDim.x) {
        __syncthreads();
        if (t < colTiles) {
            const int rt = t % mt, ct = t / mt;
            for (int r = tid; r < BM; r += 256) {
                const int i = rt * BM + r;
                rowIdx[r] = i < p.mc ? i : -1;
            }
            for (int c = tid; c < BN; c += 256) {
                const int li = ct * BN + c;
                int j = -1;
                if (li < cb) {
                    j = p.listB[li];
                    if (j < 0 || j >= p.n) j = -1;
                }
                colIdx[c] = j;
                colOk[c] = j >= 0;
            }
        } else {
            const int t2 = t - colTiles;
            const int ct = t2 % nt, rt = t2 / nt;
            for (int r = tid; r < BM; r += 256) {
                const int li = rt * BM + r;
                int i = -1;
                if (li < ca) {
                    i = p.listA[li];
                    if (i < 0 || i >= p.mc) i = -1;
                }
                rowIdx[r] = i;
            }
            for (int c = tid; c < BN; c += 256) {
                const int j = ct * BN + c;
                const bool ok = j < p.n;
                colIdx[c] = ok ? j : -1;
                colOk[c] = ok && p.eB[j] != VF_BF16_SENT;  // failed columns belong to column tiles
            }
        }
        __syncthreads();
        float acc[TM][TN];
#pragma unroll
        for (int a = 0; a < TM; ++a)
#pragma unroll
            for (int b = 0; b < TN; ++b) acc[a][b] = 0.0f;
        for (int l0 = 0; l0 < p.k; l0 += BK) {
            if (p.a_sl == 1) {
                for (int e = tid; e < BM * BK; e += 256) {
                    const int r = e / BK, l = e % BK, gl = l0 + l, i = rowIdx[r];
                    As[l][r] = (i >= 0 && gl < p.k) ? bf2f(__ldg(p.A + (long long)i * p.a_si + gl)) : 0.0f;
                }
            } else {
                for (int e = tid; e < BM * BK; e += 256) {
                    const int r = e % BM, l = e / BM, gl = l0 + l, i = rowIdx[r];
                    As[l][r] = (i >= 0 && gl < p.k) ? bf2f(__ldg(p.A + (long long)i * p.a_si + (long long)gl * p.a_sl)) : 0.0f;
                }
            }
            if (p.b_sl == 1) {
                for (int e = tid; e < BN * BK; e += 256) {
                    const int c = e / BK, l = e % BK, gl = l0 + l, j = colIdx[c];
                    Bs[l][c] = (j >= 0 && gl < p.k) ? bf2f(__ldg(p.B + gl + (long long)j * p.b_sj)) : 0.0f;
                }
            } else {
                for (int e = tid; e < BN * BK; e += 256) {
                    const int c = e % BN, l = e / BN, gl = l0 + l, j = colIdx[c];
                    Bs[l][c] = (j >= 0 && gl < p.k) ? bf2f(__ldg(p.B + (long long)gl * p.b_sl + (long long)j * p.b_sj)) : 0.0f;
                }
            }
            __syncthreads();
#pragma unroll
            for (int l = 0; l < BK; ++l) {
                float av[TM], bv[TN];
#pragma unroll
                for (int a = 0; a < TM; ++a) av[a] = As[l][tr * TM + a];
#pragma unroll
                for (int b = 0; b < TN; ++b) bv[b] = Bs[l][tc * TN + b];
#pragma unroll
                for (int a = 0; a < TM; ++a)
#pragma unroll
                    for (int b = 0; b < TN; ++b) acc[a][b] = CHAIN(av[a], bv[b], acc[a][b]);
            }
            __syncthreads();
        }
#pragma unroll
        for (int a = 0; a < TM; ++a)
#pragma unroll
            for (int b = 0; b < TN; ++b) {
                const int r = tr * TM + a, c = tc * TN + b;
                const int i = rowIdx[r], j = colIdx[c];
                if (i >= 0 && j >= 0 && colOk[c]) {
                    const float v = finalize(p.o, sc, __fmul_rn(sc.alpha, acc[a][b]), i, (long long)j * p.o.ldc + i);
                    wr_d(p.o, (long long)j * p.o.ldd + i, v);
                }
            }
    }
}

extern "C" __global__ void __launch_bounds__(256) vf_bf16_fallback_wide(VfFbArgs p) { fallback_body<64, 64, 16, 4, 4>(p); }

extern "C" __global__ void __launch_bounds__(256) vf_bf16_fallback_narrow(VfFbArgs p) { fallback_body<64, 8, 128, 1, 2>(p); }

// ---- fast fallback for the TN layout --------------------------------------------------------
// op(A) rows and op(B) columns contiguous along k, rows 16-byte aligned (lda, ldb multiples of 8,
// aligned bases), k a multiple of 8. Double-buffered shared memory, the next k-step's global
// loads prefetched into registers, 16-byte loads. Same arithmetic as the generic kernel: one fp32
// FMA chain per element over l = 0..k-1 (the zero padding after k leaves it unchanged), so its
// values do not depend on the tile configuration. The column-tile width follows the number of
// failed columns (read on the device), so the cost scales with the failed fraction.

template <int BM_, int BN_, int BK_, int TM_, int TN_, bool RF_>
struct FbCfg {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, TM = TM_, TN = TN_;
    static constexpr bool RF = RF_;  // consecutive threads take consecutive rows
    static constexpr int TR = BM / TM, TC = BN / TN;
    static_assert(TR * TC == 256, "256 threads");
    static constexpr int KC = BK / 8;  // 16-byte chunks per row per k-step
    static constexpr int NA = (BM * KC + 255) / 256, NB = (BN * KC + 255) / 256;
    static constexpr int SA = BM + 4, SB = BN + 4;
    static constexpr int SMEM = 2 * BK * (SA + SB) * 4 + (BM + BN) * 4 + BN;
};

using FbC1 = FbCfg<64, 32, 16, 4, 2, false>;   // 9..32 failed columns
using FbC2 = FbCfg<128, 64, 16, 8, 4, false>;  // more
using FbC3 = FbCfg<64, 64, 16, 4, 4, false>;   // more, when 128x64 tiles would not fill the GPU
static_assert(FbC1::SMEM <= VF_BF16_FB_TN_SMEM && FbC2::SMEM <= VF_BF16_FB_TN_SMEM && FbC3::SMEM <= VF_BF16_FB_TN_SMEM,
              "smem");

// k-major variant for few columns (decode, <= 8 failed columns) and for the failed rows of op(A):
// one output per thread (32 rows x 8 columns), tiles stored row by row along k in shared memory,
// so that 16-byte global loads become 16-byte shared stores and the operands are read as float4
// (4 consecutive k per load, still accumulated one after the other).
struct FbK {
    static constexpr int BM = 32, BN = 8, BK = 128;
    static constexpr int KC = BK / 8;
    static constexpr int SK = BK + 4;  // row stride in floats (16-byte multiple)
    static constexpr int NA = BM * KC / 256, NB = (BN * KC + 255) / 256;
    static constexpr int SMEM = 2 * (BM + BN) * SK * 4 + (BM + BN) * 4 + BN;
};
static_assert(FbK::SMEM <= VF_BF16_FB_TN_SMEM, "smem");

// Position in the tile of the a-th row (column) of thread t: groups of 4 contiguous, so that the
// operand reads are 16-byte shared loads.
template <int T, int B>
static __device__ __forceinline__ int tpos(int t, int a)
{
    if constexpr (T >= 4)
        return (a / 4) * (B / (T / 4)) + t * 4 + (a % 4);
    else
        return t * T + a;
}

static __device__ __forceinline__ void unpack8(const uint4 &w, float *dst, int stride)
{
    dst[0 * stride] = bf2f(w.x & 0xffffu);
    dst[1 * stride] = bf2f(w.x >> 16);
    dst[2 * stride] = bf2f(w.y & 0xffffu);
    dst[3 * stride] = bf2f(w.y >> 16);
    dst[4 * stride] = bf2f(w.z & 0xffffu);
    dst[5 * stride] = bf2f(w.z >> 16);
    dst[6 * stride] = bf2f(w.w & 0xffffu);
    dst[7 * stride] = bf2f(w.w >> 16);
}

template <class C>
static __device__ __forceinline__ void fb_tile(const VfFbArgs &p, const Scal &sc, unsigned char *smem, bool rows_list,
                                               int r0, bool cols_list, int c0, int ca, int cb)
{
    float *As = reinterpret_cast<float *>(smem);
    float *Bs = As + 2 * C::BK * C::SA;
    int *rowIdx = reinterpret_cast<int *>(Bs + 2 * C::BK * C::SB);
    int *colIdx = rowIdx + C::BM;
    unsigned char *colOk = reinterpret_cast<unsigned char *>(colIdx + C::BN);
    const int tid = threadIdx.x;
    __syncthreads();  // the previous tile no longer uses the shared arrays
    for (int r = tid; r < C::BM; r += 256) {
        int i = -1;
        if (rows_list) {
            const int li = r0 + r;
            if (li < ca) {
                i = p.listA[li];
                if (i < 0 || i >= p.mc) i = -1;
            }
        } else if (r0 + r < p.mc)
            i = r0 + r;
        rowIdx[r] = i;
    }
    for (int c = tid; c < C::BN; c += 256) {
        int j = -1;
        bool ok;
        if (cols_list) {
            const int li = c0 + c;
            if (li < cb) {
                j = p.listB[li];
                if (j < 0 || j >= p.n) j = -1;
            }
            ok = j >= 0;
        } else {
            if (c0 + c < p.n) j = c0 + c;
            ok = j >= 0 && p.eB[j] != VF_BF16_SENT;  // failed columns belong to column tiles
        }
        colIdx[c] = j;
        colOk[c] = ok;
    }
    __syncthreads();
    const uint4 *pa[C::NA], *pb[C::NB];
    int oa[C::NA], ob[C::NB], ka[C::NA], kb[C::NB];
    bool va[C::NA], vb[C::NB], sa_[C::NA], sb_[C::NB];
#pragma unroll
    for (int q = 0; q < C::NA; ++q) {
        const int e = tid + 256 * q;
        sa_[q] = e < C::BM * C::KC;
        const int r = e / C::KC, kc = e % C::KC;
        const int i = sa_[q] ? rowIdx[r] : -1;
        va[q] = i >= 0;
        pa[q] = reinterpret_cast<const uint4 *>(p.A + (long long)(i >= 0 ? i : 0) * p.a_si) + kc;
        oa[q] = kc * 8 * C::SA + r;
        ka[q] = kc * 8;
    }
#pragma unroll
    for (int q = 0; q < C::NB; ++q) {
        const int e = tid + 256 * q;
        sb_[q] = e < C::BN * C::KC;
        const int c = e / C::KC, kc = e % C::KC;
        const int j = sb_[q] ? colIdx[c] : -1;
        vb[q] = j >= 0;
        pb[q] = reinterpret_cast<const uint4 *>(p.B + (long long)(j >= 0 ? j : 0) * p.b_sj) + kc;
        ob[q] = kc * 8 * C::SB + c;
        kb[q] = kc * 8;
    }
    uint4 ra[C::NA], rb[C::NB];
    const uint4 z = zero4();
    auto gload = [&](int s) {
        const int l0 = s * C::BK;
#pragma unroll
        for (int q = 0; q < C::NA; ++q) ra[q] = (va[q] && l0 + ka[q] < p.k) ? __ldg(pa[q] + s * C::KC) : z;
#pragma unroll
        for (int q = 0; q < C::NB; ++q) rb[q] = (vb[q] && l0 + kb[q] < p.k) ? __ldg(pb[q] + s * C::KC) : z;
    };
    auto sstore = [&](int buf) {
#pragma unroll
        for (int q = 0; q < C::NA; ++q)
            if (sa_[q]) unpack8(ra[q], As + buf * C::BK * C::SA + oa[q], C::SA);
#pragma unroll
        for (int q = 0; q < C::NB; ++q)
            if (sb_[q]) unpack8(rb[q], Bs + buf * C::BK * C::SB + ob[q], C::SB);
    };
    int ty, tx;
    if (C::RF) {
        ty = tid % C::TR;
        tx = tid / C::TR;
    } else {
        tx = tid % C::TC;
        ty = tid / C::TC;
    }
    float acc[C::TM][C::TN];
#pragma unroll
    for (int a = 0; a < C::TM; ++a)
#pragma unroll
        for (int b = 0; b < C::TN; ++b) acc[a][b] = 0.0f;
    const int nsteps = (p.k + C::BK - 1) / C::BK;
    gload(0);
    sstore(0);
    __syncthreads();
    for (int s = 0; s < nsteps; ++s) {
        const int cur = s & 1;
        if (s + 1 < nsteps) gload(s + 1);
        const float *Ac = As + cur * C::BK * C::SA;
        const float *Bc = Bs + cur * C::BK * C::SB;
#pragma unroll
        for (int l = 0; l < C::BK; ++l) {
            float av[C::TM], bv[C::TN];
            if constexpr (C::TM >= 4) {
#pragma unroll
                for (int g = 0; g < C::TM / 4; ++g) {
                    const float4 t4 = *reinterpret_cast<const float4 *>(Ac + l * C::SA + tpos<C::TM, C::BM>(ty, 4 * g));
                    av[4 * g] = t4.x, av[4 * g + 1] = t4.y, av[4 * g + 2] = t4.z, av[4 * g + 3] = t4.w;
                }
            } else {
#pragma unroll
                for (int a = 0; a < C::TM; ++a) av[a] = Ac[l * C::SA + tpos<C::TM, C::BM>(ty, a)];
            }
            if constexpr (C::TN >= 4) {
#pragma unroll
                for (int g = 0; g < C::TN / 4; ++g) {
                    const float4 t4 = *reinterpret_cast<const float4 *>(Bc + l * C::SB + tpos<C::TN, C::BN>(tx, 4 * g));
                    bv[4 * g] = t4.x, bv[4 * g + 1] = t4.y, bv[4 * g + 2] = t4.z, bv[4 * g + 3] = t4.w;
                }
            } else {
#pragma unroll
                for (int b = 0; b < C::TN; ++b) bv[b] = Bc[l * C::SB + tpos<C::TN, C::BN>(tx, b)];
            }
#pragma unroll
            for (int a = 0; a < C::TM; ++a)
#pragma unroll
                for (int b = 0; b < C::TN; ++b) acc[a][b] = CHAIN(av[a], bv[b], acc[a][b]);
        }
        if (s + 1 < nsteps) sstore(cur ^ 1);
        __syncthreads();
    }
#pragma unroll
    for (int a = 0; a < C::TM; ++a) {
        const int i = rowIdx[tpos<C::TM, C::BM>(ty, a)];
        if (i < 0) continue;
#pragma unroll
        for (int b = 0; b < C::TN; ++b) {
            const int c = tpos<C::TN, C::BN>(tx, b);
            if (!colOk[c]) continue;
            const int j = colIdx[c];
            const float v = finalize(p.o, sc, __fmul_rn(sc.alpha, acc[a][b]), i, (long long)j * p.o.ldc + i);
            wr_d(p.o, (long long)j * p.o.ldd + i, v);
        }
    }
}

static __device__ __forceinline__ void unpack8v(const uint4 &w, float *dst)
{
    float4 lo, hi;
    lo.x = bf2f(w.x & 0xffffu), lo.y = bf2f(w.x >> 16), lo.z = bf2f(w.y & 0xffffu), lo.w = bf2f(w.y >> 16);
    hi.x = bf2f(w.z & 0xffffu), hi.y = bf2f(w.z >> 16), hi.z = bf2f(w.w & 0xffffu), hi.w = bf2f(w.w >> 16);
    reinterpret_cast<float4 *>(dst)[0] = lo;
    reinterpret_cast<float4 *>(dst)[1] = hi;
}

static __device__ __forceinline__ void fb_tile_k(const VfFbArgs &p, const Scal &sc, unsigned char *smem,
                                                 bool rows_list, int r0, bool cols_list, int c0, int ca, int cb)
{
    using C = FbK;
    float *As = reinterpret_cast<float *>(smem);       // [2][BM][SK]
    float *Bs = As + 2 * C::BM * C::SK;                 // [2][BN][SK]
    int *rowIdx = reinterpret_cast<int *>(Bs + 2 * C::BN * C::SK);
    int *colIdx = rowIdx + C::BM;
    unsigned char *colOk = reinterpret_cast<unsigned char *>(colIdx + C::BN);
    const int tid = threadIdx.x;
    __syncthreads();
    if (tid < C::BM) {
        int i = -1;
        if (rows_list) {
            const int li = r0 + tid;
            if (li < ca) {
                i = p.listA[li];
                if (i < 0 || i >= p.mc) i = -1;
            }
        } else if (r0 + tid < p.mc)
            i = r0 + tid;
        rowIdx[tid] = i;
    } else if (tid < C::BM + C::BN) {
        const int c = tid - C::BM;
        int j = -1;
        bool ok;
        if (cols_list) {
            const int li = c0 + c;
            if (li < cb) {
                j = p.listB[li];
                if (j < 0 || j >= p.n) j = -1;
            }
            ok = j >= 0;
        } else {
            if (c0 + c < p.n) j = c0 + c;
            ok = j >= 0 && p.eB[j] != VF_BF16_SENT;
        }
        colIdx[c] = j;
        colOk[c] = ok;
    }
    __syncthreads();
    const uint4 *pa[C::NA], *pb[C::NB];
    int oa[C::NA], ob[C::NB], ka[C::NA], kb[C::NB];
    bool va[C::NA], vb[C::NB], sb_[C::NB];
#pragma unroll
    for (int q = 0; q < C::NA; ++q) {
        const int e = tid + 256 * q, r = e / C::KC, kc = e % C::KC, i = rowIdx[r];
        va[q] = i >= 0;
        pa[q] = reinterpret_cast<const uint4 *>(p.A + (long long)(i >= 0 ? i : 0) * p.a_si) + kc;
        oa[q] = r * C::SK + kc * 8;
        ka[q] = kc * 8;
    }
#pragma unroll
    for (int q = 0; q < C::NB; ++q) {
        const int e = tid + 256 * q;
        sb_[q] = e < C::BN * C::KC;
        const int c = e / C::KC, kc = e % C::KC, j = sb_[q] ? colIdx[c] : -1;
        vb[q] = j >= 0;
        pb[q] = reinterpret_cast<const uint4 *>(p.B + (long long)(j >= 0 ? j : 0) * p.b_sj) + kc;
        ob[q] = c * C::SK + kc * 8;
        kb[q] = kc * 8;
    }
    uint4 ra[C::NA], rb[C::NB];
    const uint4 z = zero4();
    auto gload = [&](int s) {
        const int l0 = s * C::BK;
#pragma unroll
        for (int q = 0; q < C::NA; ++q) ra[q] = (va[q] && l0 + ka[q] < p.k) ? __ldg(pa[q] + s * C::KC) : z;
#pragma unroll
        for (int q = 0; q < C::NB; ++q) rb[q] = (vb[q] && l0 + kb[q] < p.k) ? __ldg(pb[q] + s * C::KC) : z;
    };
    auto sstore = [&](int buf) {
#pragma unroll
        for (int q = 0; q < C::NA; ++q) unpack8v(ra[q], As + buf * C::BM * C::SK + oa[q]);
#pragma unroll
        for (int q = 0; q < C::NB; ++q)
            if (sb_[q]) unpack8v(rb[q], Bs + buf * C::BN * C::SK + ob[q]);
    };
    const int ty = tid % C::BM, tx = tid / C::BM;  // a warp: 32 rows, one column
    float acc = 0.0f;
    const int nsteps = (p.k + C::BK - 1) / C::BK;
    gload(0);
    sstore(0);
    __syncthreads();
    for (int s = 0; s < nsteps; ++s) {
        const int cur = s & 1;
        if (s + 1 < nsteps) gload(s + 1);
        const float4 *a4 = reinterpret_cast<const float4 *>(As + cur * C::BM * C::SK + ty * C::SK);
        const float4 *b4 = reinterpret_cast<const float4 *>(Bs + cur * C::BN * C::SK + tx * C::SK);
#pragma unroll
        for (int l = 0; l < C::BK / 4; ++l) {
            const float4 a = a4[l], b = b4[l];
            acc = CHAIN(a.x, b.x, acc);
            acc = CHAIN(a.y, b.y, acc);
            acc = CHAIN(a.z, b.z, acc);
            acc = CHAIN(a.w, b.w, acc);
        }
        if (s + 1 < nsteps) sstore(cur ^ 1);
        __syncthreads();
    }
    const int i = rowIdx[ty];
    if (i >= 0 && colOk[tx]) {
        const int j = colIdx[tx];
        const float v = finalize(p.o, sc, __fmul_rn(sc.alpha, acc), i, (long long)j * p.o.ldc + i);
        wr_d(p.o, (long long)j * p.o.ldd + i, v);
    }
}

extern "C" __global__ void __launch_bounds__(256, 2) vf_bf16_fallback_tn(VfFbArgs p)
{
    extern __shared__ __align__(16) unsigned char smem[];
    const int ca = imin32(imax32(*p.cntA, 0), p.mc);
    const int cb = imin32(imax32(*p.cntB, 0), p.n);
    // Column tiles: 128x64 when they fill the persistent grid twice, else 64x64; 64x32 for 9..32
    // failed columns; the k-major 32x8 tiles for up to 8.
    int cfg = cb > 32 ? 2 : (cb > 8 ? 1 : 0);
    if (cfg == 2 && cdiv(cb, FbC2::BN) * cdiv(p.mc, FbC2::BM) < 2 * (int)gridDim.x) cfg = 3;
    const int mtile = cfg == 2 ? FbC2::BM : cfg == 3 ? FbC3::BM : cfg == 1 ? FbC1::BM : FbK::BM;
    const int ntile = cfg == 2 ? FbC2::BN : cfg == 3 ? FbC3::BN : cfg == 1 ? FbC1::BN : FbK::BN;
    const int mt = cdiv(p.mc, mtile);
    const int colTiles = cdiv(cb, ntile) * mt;
    const int nt0 = cdiv(p.n, FbK::BN);
    const int total = colTiles + cdiv(ca, FbK::BM) * nt0;
    if ((int)blockIdx.x >= total) return;
    const Scal sc = scalars(p.o);
    for (int t = blockIdx.x; t < total; t += gridDim.x) {
        if (t < colTiles) {
            const int r0 = (t % mt) * mtile, c0 = (t / mt) * ntile;
            if (cfg == 2)
                fb_tile<FbC2>(p, sc, smem, false, r0, true, c0, ca, cb);
            else if (cfg == 3)
                fb_tile<FbC3>(p, sc, smem, false, r0, true, c0, ca, cb);
            else if (cfg == 1)
                fb_tile<FbC1>(p, sc, smem, false, r0, true, c0, ca, cb);
            else
                fb_tile_k(p, sc, smem, false, r0, true, c0, ca, cb);
        } else {
            const int t2 = t - colTiles;
            fb_tile_k(p, sc, smem, true, (t2 / nt0) * FbK::BM, false, (t2 % nt0) * FbK::BN, ca, cb);
        }
    }
}

// ---- the skinny path, workspace-free -----------------------------------------------------------
// C(i, j) = alpha * sum_{l=0}^{k-1} op(A)(i, l) op(B)(l, j) (+ beta C + bias): one fp32 FMA chain per
// element over l = 0, 1, ..., k-1 from +0, the fallback's arithmetic, so its values do not depend
// on the tile configuration. A CTA takes RB rows of op(A) and all (<= NCP) columns of op(B): every
// row of A is read from memory once. One output per thread; operands staged in shared memory as
// fp32 (converted once, when stored), double-buffered, the next k-step prefetched into registers.
// FAST: rows of op(A) and columns of op(B) contiguous along k, 16-byte aligned, k a multiple of 8
// (16-byte loads); else element loads (same values).
template <int NCP, int RB, bool FAST>
static __device__ __forceinline__ void skinny_body(const VfSkArgs &p)
{
    constexpr int T = NCP * RB;
    constexpr int BK = 128;
    constexpr int SK = BK + 4;            // floats per shared row: 16-byte multiple, 4 mod 32 banks
    constexpr int KC = BK / 8;            // 16-byte chunks per row per k-step
    constexpr int CA = RB * KC, CB = NCP * KC;
    constexpr int NA = (CA + T - 1) / T, NB = (CB + T - 1) / T;
    __shared__ __align__(16) float As[2][RB][SK];
    __shared__ __align__(16) float Bs[2][NCP][SK];
    const int tid = threadIdx.x;
    const int c = tid % NCP, r = tid / NCP;
    const int i0 = blockIdx.x * RB;
    uint4 ra[NA], rb[NB];
    const uint4 z = zero4();
    auto gload = [&](int s) {
        const int l0 = s * BK;
#pragma unroll
        for (int q = 0; q < NA; ++q) {
            const int e = tid + q * T;
            uint4 v = z;
            if (e < CA) {
                const int row = e / KC, kc = e % KC, i = i0 + row, l = l0 + kc * 8;
                if (i < p.m) {
                    if (FAST) {
                        if (l < p.k) v = __ldg(reinterpret_cast<const uint4 *>(p.A + (long long)i * p.a_si + l));
                    } else {
                        unsigned short h[8];
#pragma unroll
                        for (int t = 0; t < 8; ++t)
                            h[t] = (l + t < p.k) ? __ldg(p.A + (long long)i * p.a_si + (long long)(l + t) * p.a_sl) : 0;
                        v.x = h[0] | ((unsigned)h[1] << 16), v.y = h[2] | ((unsigned)h[3] << 16);
                        v.z = h[4] | ((unsigned)h[5] << 16), v.w = h[6] | ((unsigned)h[7] << 16);
                    }
                }
            }
            ra[q] = v;
        }
#pragma unroll
        for (int q = 0; q < NB; ++q) {
            const int e = tid + q * T;
            uint4 v = z;
            if (e < CB) {
                const int col = e / KC, kc = e % KC, l = l0 + kc * 8;
                if (col < p.n) {
                    if (FAST) {
                        if (l < p.k) v = __ldg(reinterpret_cast<const uint4 *>(p.B + l + (long long)col * p.b_sj));
                    } else {
                        unsigned short h[8];
#pragma unroll
                        for (int t = 0; t < 8; ++t)
                            h[t] = (l + t < p.k) ? __ldg(p.B + (long long)(l + t) * p.b_sl + (long long)col * p.b_sj) : 0;
                        v.x = h[0] | ((unsigned)h[1] << 16), v.y = h[2] | ((unsigned)h[3] << 16);
                        v.z = h[4] | ((unsigned)h[5] << 16), v.w = h[6] | ((unsigned)h[7] << 16);
                    }
                }
            }
            rb[q] = v;
        }
    };
    auto sstore = [&](int buf) {
#pragma unroll
        for (int q = 0; q < NA; ++q) {
            const int e = tid + q * T;
            if (e < CA) unpack8v(ra[q], &As[buf][e / KC][(e % KC) * 8]);
        }
#pragma unroll
        for (int q = 0; q < NB; ++q) {
            const int e = tid + q * T;
            if (e < CB) unpack8v(rb[q], &Bs[buf][e / KC][(e % KC) * 8]);
        }
    };
    float acc = 0.0f;
#ifdef VF_KERNEL_CONTROL_SPLITK  // two chains, over even and odd k-steps, added at the end
    float acc2 = 0.0f;
#endif
    const int nsteps = (p.k + BK - 1) / BK;
    gload(0);
    sstore(0);
    __syncthreads();
    for (int s = 0; s < nsteps; ++s) {
#ifdef VF_KERNEL_CONTROL_SPLITK
        { float t = acc; acc = acc2; acc2 = t; }
#endif
        const int cur = s & 1;
        if (s + 1 < nsteps) gload(s + 1);
        const float4 *a4 = reinterpret_cast<const float4 *>(&As[cur][r][0]);
        const float4 *b4 = reinterpret_cast<const float4 *>(&Bs[cur][c][0]);
        // zero padding past k (FAST: k % 8 == 0 and whole chunks; else zeros) leaves the chain
        // unchanged: fma(0, 0, acc) == acc for every acc, and acc is never -0 from a +0 start
#pragma unroll
        for (int l4 = 0; l4 < BK / 4; ++l4) {
            const float4 a = a4[l4], b = b4[l4];
            acc = CHAIN(a.x, b.x, acc);
            acc = CHAIN(a.y, b.y, acc);
            acc = CHAIN(a.z, b.z, acc);
            acc = CHAIN(a.w, b.w, acc);
        }
        if (s + 1 < nsteps) sstore(cur ^ 1);
        __syncthreads();
    }
#ifdef VF_KERNEL_CONTROL_SPLITK
    acc = __fadd_rn(acc, acc2);
#endif
    const int i = i0 + r;
    if (i < p.m && c < p.n) {
        const Scal sc = scalars(p.o);
        const float v = finalize(p.o, sc, __fmul_rn(sc.alpha, acc), i, (long long)c * p.o.ldc + i);
        wr_d(p.o, (long long)c * p.o.ldd + i, v);
    }
}

#define VF_SKINNY(NCP, RB)                                                                                  \
    extern "C" __global__ void __launch_bounds__(NCP * RB) vf_bf16_skinny_##NCP##_##RB(VfSkArgs p)          \
    {                                                                                                       \
        skinny_body<NCP, RB, true>(p);                                                                      \
    }                                                                                                       \
    extern "C" __global__ void __launch_bounds__(NCP * RB) vf_bf16_skinny_g_##NCP##_##RB(VfSkArgs p)        \
    {                                                                                                       \
        skinny_body<NCP, RB, false>(p);                                                                     \
    }
VF_SKINNY(8, 16)
VF_SKINNY(8, 32)
VF_SKINNY(16, 8)
VF_SKINNY(16, 16)
VF_SKINNY(32, 4)
VF_SKINNY(32, 8)
