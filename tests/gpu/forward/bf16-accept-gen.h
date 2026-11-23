/* Deterministic bf16 operands for the acceptance test of the bf16 tensor-core path
 * (tests/gpu/forward/bf16-accept.c), and their exact products.
 *
 * Generator: every element is a function of (seed, stream, index) alone (a counter-based splitmix64), and is
 * built with exactly rounded double operations in a fixed order, then rounded to fp32 and to bf16 (nearest
 * even). A NumPy restatement of it gives the same bits (it ran the same data on an RTX 5090 for the native
 * bias slopes of bf16-accept.c), and an element does not depend on k (a shorter k is a prefix).
 *
 * Row i of op(A) and column j of op(B) are vectors of length k: va(g, i, l) and vb(g, j, l).
 *
 * Exact products: every bf16 value is M * 2^E with M an integer of at most 8 bits; a sum of k <= 2^20 such
 * products is kept exactly in 20 digits of 32 bits (2^-266 to 2^374) and rounded once (to double, fp32 and
 * bf16, nearest even, with subnormals and overflow). */
#ifndef BF16_ACCEPT_GEN_H
#define BF16_ACCEPT_GEN_H
#include <math.h>
#include <stdint.h>
#include <string.h>

enum { FAM_PARAM, FAM_NORMAL, FAM_SAMESIGN, FAM_LOGNPOS, FAM_ADVPOS, FAM_CRAFTSUB, FAM_CRAFTACT, FAM_RELU2,
       FAM_OUTLIER, FAM_CANCEL, FAM_MIXED30, FAM_N };
static const char *const fam_name[FAM_N] = {"param",    "normal",   "samesign", "lognpos", "advpos", "craftsub",
                                            "craftact", "relu2",    "outlier",  "cancel",  "mixed30"};

typedef struct {
    int fam;
    uint64_t seed;
    /* FAM_PARAM: signs (sA: 0 random, 1 positive; sB: 0 random, 1 positive, 2 half zeros and positive),
     * binary-exponent spreads, per-row/column scales 2^[-R, R], outliers of A (probability qA, factor 2^EA),
     * outliers of B (pos: 0 probability qB, 1 one per column at l < 256, 2 at l = 0, 3 at every multiple of
     * 256, 4 one l < 256 for all columns; factor 2^EB), cancelling pairs. */
    int sA, sB, R, EA, EB, pos, cancel;
    double sigA, sigB, qA, qB;
} Gen;

static inline uint64_t g_mix(uint64_t z)
{
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static inline uint64_t g_r(const Gen *g, uint64_t st, uint64_t idx)
{
    return g_mix((g->seed ^ (st * 0xD1B54A32D192ED03ull)) + idx * 0x9E3779B97F4A7C15ull);
}
static inline double g_u(const Gen *g, uint64_t st, uint64_t idx) { return (double)(g_r(g, st, idx) >> 40) * 0x1p-24; }
static inline double g_gauss(const Gen *g, uint64_t st, uint64_t idx)
{
    volatile double s = g_u(g, st, idx); /* volatile: the order of the sums is part of the definition */
    s = s + g_u(g, st + 1, idx);
    s = s + g_u(g, st + 2, idx);
    s = s + g_u(g, st + 3, idx);
    s = s - 2.0;
    return s * 1.7320508075688772;
}
static inline double g_sign(const Gen *g, uint64_t st, uint64_t idx) { return (g_r(g, st, idx) & 1) ? -1.0 : 1.0; }
static inline long long g_int(const Gen *g, uint64_t st, uint64_t idx, long long lo, long long hi)
{
    return lo + (long long)(g_r(g, st, idx) % (uint64_t)(hi - lo + 1));
}

static inline uint16_t g_bf16(double x)
{
    const float f = (float)x;
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fff;
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static inline double g_f(uint16_t b)
{
    const uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

#define G_IDX(i, l) (((uint64_t)(i) << 20) + (uint64_t)(l))

/* Row i of op(A), element l, before the cancelling pairs. */
static double g_a0(const Gen *g, long long i, int l)
{
    const uint64_t x = G_IDX(i, l);
    switch (g->fam) {
    case FAM_NORMAL: case FAM_CRAFTACT: case FAM_RELU2: case FAM_OUTLIER: case FAM_CANCEL:
        return 0.02 * g_gauss(g, 1, x);
    case FAM_SAMESIGN: return g_u(g, 1, x);
    case FAM_LOGNPOS: return 0.02 * (1.0 + g_u(g, 1, x)) * ldexp(1.0, (int)rint(2.0 * g_gauss(g, 2, x)));
    case FAM_ADVPOS: {
        double v = 0.02 * (1.0 + g_u(g, 1, x)) * ldexp(1.0, (int)rint(2.45 * g_gauss(g, 2, x)));
        v *= ldexp(1.0, (int)g_int(g, 6, (uint64_t)i, -4, 4));
        if (g_r(g, 7, x) % 1000 == 0) v *= 2048.0;
        return v;
    }
    case FAM_CRAFTSUB: return (l % 256 == 0) ? 1024.0 : 0.353515625 * (0.5 + 0.5 * g_u(g, 1, x));
    case FAM_MIXED30: return g_sign(g, 1, x) * (1.0 + g_u(g, 2, x)) * ldexp(1.0, (int)g_int(g, 3, x, -21, 8));
    default: { /* FAM_PARAM */
        double v = (1.0 + g_u(g, 1, x)) * ldexp(1.0, (int)rint(g->sigA * g_gauss(g, 2, x)));
        if (g->sA == 0) v *= g_sign(g, 6, x);
        v *= 0.02;
        if (g->R) v *= ldexp(1.0, (int)g_int(g, 7, (uint64_t)i, -g->R, g->R));
        if (g->qA > 0 && g_u(g, 8, x) < g->qA) v *= ldexp(1.0, g->EA);
        return v;
    }
    }
}

/* Column j of op(B), element l, before the cancelling pairs. */
static double g_b0(const Gen *g, long long j, int l)
{
    const uint64_t x = G_IDX(j, l);
    switch (g->fam) {
    case FAM_NORMAL: case FAM_CANCEL: return g_gauss(g, 21, x);
    case FAM_SAMESIGN: return g_u(g, 21, x);
    case FAM_LOGNPOS: return (1.0 + g_u(g, 21, x)) * ldexp(1.0, (int)rint(2.0 * g_gauss(g, 22, x)));
    case FAM_ADVPOS: {
        double v = (1.0 + g_u(g, 21, x)) * ldexp(1.0, (int)rint(2.04 * g_gauss(g, 22, x)));
        v *= ldexp(1.0, (int)g_int(g, 26, (uint64_t)j, -4, 4));
        if (l % 256 == 0) v *= 8.0;
        return v;
    }
    case FAM_CRAFTSUB: return (l % 256 == 0) ? 1024.0 : 0.353515625;
    case FAM_CRAFTACT: return (l % 256 == 0) ? 2048.0 * g_sign(g, 26, x) : g_gauss(g, 21, x);
    case FAM_RELU2: {
        const double z = g_gauss(g, 21, x);
        return z > 0 ? 0.1 * (z * z) : 0.0;
    }
    case FAM_OUTLIER:
        return (l == (int)g_int(g, 26, (uint64_t)j, 0, 255)) ? g_sign(g, 27, x) * 2048.0 * (1.0 + 0.5 * g_u(g, 28, x))
                                                            : g_gauss(g, 21, x);
    case FAM_MIXED30: return g_sign(g, 21, x) * (1.0 + g_u(g, 22, x)) * ldexp(1.0, (int)g_int(g, 23, x, -15, 14));
    default: { /* FAM_PARAM */
        if (g->sB == 2 && (g_r(g, 26, x) & 1)) return 0.0;
        double v = (1.0 + g_u(g, 21, x)) * ldexp(1.0, (int)rint(g->sigB * g_gauss(g, 22, x)));
        if (g->sB == 0) v *= g_sign(g, 27, x);
        if (g->R) v *= ldexp(1.0, (int)g_int(g, 28, (uint64_t)j, -g->R, g->R));
        int out = 0;
        switch (g->pos) {
        case 0: out = g->qB > 0 && g_u(g, 29, x) < g->qB; break;
        case 1: out = l == (int)g_int(g, 30, (uint64_t)j, 0, 255); break;
        case 2: out = l == 0; break;
        case 3: out = l % 256 == 0; break;
        default: out = l == (int)g_int(g, 30, 0, 0, 255); break;
        }
        if (out) v = (v == 0.0 ? 1.0 : v) * ldexp(1.0, g->EB);
        return v;
    }
    }
}

static inline int g_cancels(const Gen *g) { return g->fam == FAM_CANCEL || (g->fam == FAM_PARAM && g->cancel); }

/* The bf16 bits of op(A)[i][l] and op(B)[l][j]. Cancelling pairs: an odd l takes -A[l-1] and
 * B[l-1] (1 + 2^-7 g). */
static uint16_t va(const Gen *g, long long i, int l)
{
    if (g_cancels(g) && (l & 1)) return (uint16_t)(va(g, i, l - 1) ^ 0x8000u);
    return g_bf16(g_a0(g, i, l));
}
static uint16_t vb(const Gen *g, long long j, int l)
{
    if (g_cancels(g) && (l & 1)) return g_bf16(g_f(vb(g, j, l - 1)) * (1.0 + 0x1p-7 * g_gauss(g, 40, G_IDX(j, l))));
    return g_bf16(g_b0(g, j, l));
}

/* ---- exact products ------------------------------------------------------------------------- */
#define EX_NDIG 20
#define EX_OFF 266

typedef struct {
    uint32_t d[EX_NDIG];
    int neg, zero;
} ExMag;

static inline void ex_dec(uint16_t b, int32_t *M, int32_t *E)
{
    const int e = (b >> 7) & 0xff, f = b & 0x7f;
    const int32_t m = e ? (f | 0x80) : f;
    *E = (e ? e : 1) - 1; /* E + 133 */
    *M = (b & 0x8000) ? -m : m;
}

static void ex_norm(const int64_t *acc, ExMag *g)
{
    int64_t a[EX_NDIG], carry = 0;
    for (int i = 0; i < EX_NDIG; ++i) {
        const int64_t v = acc[i] + carry;
        a[i] = v & 0xffffffffLL;
        carry = v >> 32;
    }
    g->neg = carry < 0;
    uint64_t c = 1;
    for (int i = 0; i < EX_NDIG; ++i) {
        if (g->neg) {
            const uint64_t v = (uint64_t)(~(uint32_t)a[i]) + c;
            g->d[i] = (uint32_t)v;
            c = v >> 32;
        } else {
            g->d[i] = (uint32_t)a[i];
        }
    }
    g->zero = 1;
    for (int i = 0; i < EX_NDIG; ++i)
        if (g->d[i]) g->zero = 0;
}
static inline int ex_bit(const ExMag *g, int p) { return p >= 0 && p < EX_NDIG * 32 ? (g->d[p >> 5] >> (p & 31)) & 1 : 0; }
static double ex_round(const ExMag *g, int p, int lsb)
{
    if (g->zero) return 0.0;
    int P = -1;
    for (int i = EX_NDIG - 1; i >= 0 && P < 0; --i)
        if (g->d[i]) P = i * 32 + 31 - __builtin_clz(g->d[i]);
    int r = P - p + 1;
    if (r < lsb + EX_OFF) r = lsb + EX_OFF;
    uint64_t q = 0;
    if (r <= 0) {
        for (int i = P; i >= 0; --i) q = (q << 1) | (uint64_t)ex_bit(g, i);
        return ldexp((double)q, -EX_OFF);
    }
    for (int i = P; i >= r; --i) q = (q << 1) | (uint64_t)ex_bit(g, i);
    const int half = ex_bit(g, r - 1);
    int sticky = 0;
    const int full = (r - 1) >> 5;
    for (int i = 0; i < full && i < EX_NDIG; ++i)
        if (g->d[i]) sticky = 1;
    if (((r - 1) & 31) && full < EX_NDIG && (g->d[full] & ((1u << ((r - 1) & 31)) - 1u))) sticky = 1;
    if (half && (sticky || (q & 1))) ++q;
    return ldexp((double)q, r - EX_OFF);
}

typedef struct {
    double x;      /* exact value rounded to nearest double */
    double s;      /* sum |a b| */
    float cr32;    /* correctly rounded fp32 */
    uint16_t cr16; /* correctly rounded bf16 */
    int nonfinite; /* an operand is inf or NaN */
} Exact;

/* The exact dot product of two bf16 vectors of length k. */
static Exact ex_dot(const uint16_t *a, const uint16_t *b, int k)
{
    int64_t acc[EX_NDIG];
    memset(acc, 0, sizeof acc);
    Exact e;
    memset(&e, 0, sizeof e);
    double s = 0;
    for (int l = 0; l < k; ++l) {
        if ((a[l] & 0x7f80) == 0x7f80 || (b[l] & 0x7f80) == 0x7f80) e.nonfinite = 1;
        int32_t Ma, Ea, Mb, Eb;
        ex_dec(a[l], &Ma, &Ea);
        ex_dec(b[l], &Mb, &Eb);
        const int64_t pr = (int64_t)(Ma * Mb);
        const int ee = Ea + Eb;
        const int64_t v = pr * ((int64_t)1 << (ee & 31));
        acc[ee >> 5] += v & 0xffffffffLL;
        acc[(ee >> 5) + 1] += v >> 32;
        s += fabs(g_f(a[l]) * g_f(b[l]));
    }
    ExMag g;
    ex_norm(acc, &g);
    const double sg = g.neg ? -1.0 : 1.0;
    e.x = e.nonfinite ? NAN : sg * ex_round(&g, 53, -1074);
    e.cr32 = (float)(sg * ex_round(&g, 24, -149));
    const float f16 = (float)(sg * ex_round(&g, 8, -133));
    uint32_t u;
    memcpy(&u, &f16, 4);
    e.cr16 = (uint16_t)(u >> 16);
    e.s = s;
    return e;
}

/* The tensor-core path's strict check (crates/vf-fwdgen/bf16tc/kernels.cu, classify and conv1), restated:
 * 1 when the vector goes to the fallback. */
static int tc_fails(const uint16_t *v, int k)
{
    unsigned mx = 0;
    for (int l = 0; l < k; ++l)
        if ((v[l] & 0x7fffu) > mx) mx = v[l] & 0x7fffu;
    if (mx >= 0x7f80u) return 1;
    if (mx == 0) return 0;
    const int be = (int)(mx >> 7);
    if (be == 0) return 1;
    const int e = be - 127;
    if (e < -24 || e > 31) return 1;
    for (int l = 0; l < k; ++l) {
        const unsigned a = v[l] & 0x7fffu;
        if (!a) continue;
        const int b2 = (int)(a >> 7);
        const unsigned M = b2 ? ((a & 0x7f) | 0x80) : (a & 0x7f);
        if ((b2 ? b2 : 1) - 134 + 15 - e + __builtin_ctz(M) < -24) return 1;
    }
    return 0;
}
#endif
