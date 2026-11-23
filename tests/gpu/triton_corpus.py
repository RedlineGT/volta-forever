"""Triton kernels and their cases, for tests/gpu/triton-diff.py.

Every case is a kernel, a grid, its arguments (tensors made on the CPU from a seed, once:
tests/gpu/triton-diff.py saves them with the reference, for both routes) and how its result is
checked against a reference: Triton's interpreter on the CPU (TRITON_INTERPRET=1) by
default, a computation with PyTorch on the CPU where the interpreter cannot run the kernel or
computes something else than the kernel says (named in the case), or nothing but the other runs
on the GPU (determinism, the other route) where neither can.

The data of most cases are chosen so that the right result is one value whatever the order of the
operations: floating-point inputs that are small integers make sums, products and their fused
forms exact, so that the result is compared bit for bit with the interpreter's although Triton
does not fix the order of a sum. Cases on data drawn at random are compared within a stated
tolerance, and bit for bit between the runs and between the two routes of the layer.

`cases(level)` gives the list: level 'quick' is the subset that tests/gpu/triton.sh runs, 'full'
everything. The module is imported both on the GPU and under the interpreter; it must not
touch a GPU at import.
"""
import math
import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------------------------------------
# The case and its data

SENTINEL = {
    torch.float16: -1234.0, torch.bfloat16: -1232.0, torch.float32: -12345.678, torch.float64: -98765.4321,
    torch.int8: -77, torch.int16: -7777, torch.int32: -777777, torch.int64: -7777777777,
    torch.uint8: 177, torch.uint16: 57777, torch.uint32: 3777777777, torch.uint64: 17777777777777,
    torch.bool: True,
}
FLOATS = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
SINTS = [torch.int8, torch.int16, torch.int32, torch.int64]
UINTS = [torch.uint8, torch.uint16, torch.uint32, torch.uint64]
INTS = SINTS + UINTS
ALL = INTS + FLOATS
SHORT = {torch.float16: 'f16', torch.bfloat16: 'bf16', torch.float32: 'f32', torch.float64: 'f64',
         torch.int8: 'i8', torch.int16: 'i16', torch.int32: 'i32', torch.int64: 'i64',
         torch.uint8: 'u8', torch.uint16: 'u16', torch.uint32: 'u32', torch.uint64: 'u64', torch.bool: 'b'}


def bits(dtype):
    return torch.finfo(dtype).bits if dtype.is_floating_point else torch.iinfo(dtype).bits


class Case:
    """One kernel launch and how its result is checked.

    make(g) -> {name: CPU tensor}: the inputs, and the outputs filled with a sentinel (so that a
        word the kernel should not write, and one it should and does not, are both seen).
    args: the kernel's arguments in order: a name of `make`'s dict, ('view', name, offset) for that
        tensor from element `offset` on (an address that is not aligned), or a Python scalar.
    outs: the names compared.
    cmp: 'exact' (bit for bit; NaN equals NaN) or ('tol', rtol, atol).
    ref: 'interp' (the interpreter), 'interp32' (the interpreter on float32 copies of the bfloat16
        tensors, the outputs rounded to bfloat16 to nearest even after: the interpreter does not
        compute in bfloat16 as the GPU does), a function(t) -> {name: tensor} computed with PyTorch
        on the CPU, or None (no reference: the runs and the routes are compared only).
    post: function(outs) -> outs applied before any comparison (to sort what atomics order).
    nondet: the runs may differ (a sum in an order that atomics choose): no determinism check.
    route_exact: the two routes of the layer must give the same bits (False only where the PTX
        allows the assembler a choice that changes bits, said in the case).
    """

    def __init__(self, name, kernel, grid, args, make, outs, meta=None, cmp='exact', ref='interp',
                 post=None, nondet=False, route_exact=True, quick=False, launch=None, note=''):
        self.name, self.kernel, self.grid, self.args = name, kernel, grid, args
        self.make, self.outs, self.meta = make, outs, dict(meta or {})
        self.cmp, self.ref, self.post, self.nondet = cmp, ref, post, nondet
        self.route_exact, self.quick, self.launch, self.note = route_exact, quick, launch, note

    def run(self, t, interpreted=False):
        """Launch on the tensors t (on the device where they are)."""
        if self.launch is not None:
            return self.launch(t, interpreted)
        args = []
        for a in self.args:
            if isinstance(a, str):
                args.append(t[a])
            elif isinstance(a, tuple) and a and a[0] == 'view':
                args.append(t[a[1]][a[2]:])
            else:
                args.append(a)
        meta = dict(self.meta)
        if interpreted:
            for k in ('num_warps', 'num_stages', 'num_ctas', 'maxnreg'):
                meta.pop(k, None)
        grid = self.grid(t) if callable(self.grid) else self.grid
        self.kernel[grid](*args, **meta)


def gen(dtype, shape, kind, g):
    """Data of `dtype` and `shape`: kind 'int' small integers, 'pos' small positive integers,
    'nz' nonzero small integers, 'rnd' normal, 'wide' normal times powers of two over the range,
    'special' normal with infinities, NaN, zeros of both signs and subnormals, 'full' every bit
    pattern (integers), 'shift' shift counts below the width, 'unit' in [0, 1)."""
    shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
    n = math.prod(shape)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, generator=g).bool()
    if dtype.is_floating_point:
        if kind == 'int':
            v = torch.randint(-8, 9, shape, generator=g).double()
        elif kind == 'small':
            v = torch.randint(-2, 3, shape, generator=g).double()
        elif kind == 'pos':
            v = torch.randint(1, 9, shape, generator=g).double()
        elif kind == 'nz':
            v = torch.randint(1, 9, shape, generator=g).double() * (torch.randint(0, 2, shape, generator=g) * 2 - 1)
        elif kind == 'rnd':
            v = torch.randn(shape, generator=g, dtype=torch.float64)
        elif kind == 'unit':
            v = torch.rand(shape, generator=g, dtype=torch.float64)
        elif kind == 'wide':
            e = {16: 14, 32: 100, 64: 600}[bits(dtype)] if dtype != torch.bfloat16 else 100
            v = torch.randn(shape, generator=g, dtype=torch.float64) * torch.pow(
                2.0, torch.randint(-e, e + 1, shape, generator=g).double())
        elif kind in ('special', 'subn'):
            v = torch.randn(shape, generator=g, dtype=torch.float64).to(dtype)
            fi = torch.finfo(dtype)
            specials = [float('inf'), float('-inf'), float('nan'), 0.0, -0.0, fi.tiny, -fi.tiny, fi.max, -fi.max,
                        1.0, -1.0, 0.5]
            if kind == 'subn':
                specials += [fi.tiny / 4, -fi.tiny / 8, fi.tiny * fi.eps]
            specials = torch.tensor(specials, dtype=torch.float64).to(dtype)
            pick = torch.randint(0, 4, shape, generator=g) == 0
            which = torch.randint(0, len(specials), shape, generator=g)
            return torch.where(pick, specials[which], v)
        else:
            raise ValueError(kind)
        return v.to(dtype)
    info = torch.iinfo(dtype)
    if kind == 'full':
        if info.bits == 64:
            v = torch.randint(-2**63, 2**63 - 1, shape, generator=g, dtype=torch.int64)
            return v.view(dtype) if dtype == torch.uint64 else v
        return torch.randint(info.min, info.max + 1, shape, generator=g, dtype=torch.int64).to(dtype)
    if kind == 'int':
        lo, hi = (0, 17) if info.min == 0 else (-8, 9)
    elif kind in ('pos', 'nz'):
        lo, hi = 1, 9
    elif kind == 'small':
        lo, hi = (0, 3) if info.min == 0 else (-2, 3)
    elif kind == 'shift':
        lo, hi = 0, info.bits
    elif kind == 'mid':
        # values whose sums and products of two stay in range
        h = 2 ** (info.bits // 2 - 1)
        lo, hi = (0, 2 * h) if info.min == 0 else (-h, h)
    else:
        raise ValueError(kind)
    v = torch.randint(lo, hi, shape, generator=g, dtype=torch.int64)
    if kind == 'nz' and info.min < 0:
        v = v * (torch.randint(0, 2, shape, generator=g, dtype=torch.int64) * 2 - 1)
    return v.to(dtype)


def sentinel(dtype, shape):
    return torch.full(tuple(shape) if isinstance(shape, (tuple, list)) else (shape,), SENTINEL[dtype], dtype=dtype)


def tl_dtype(dtype):
    return {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32,
            torch.float64: tl.float64, torch.int8: tl.int8, torch.int16: tl.int16, torch.int32: tl.int32,
            torch.int64: tl.int64, torch.uint8: tl.uint8, torch.uint16: tl.uint16, torch.uint32: tl.uint32,
            torch.uint64: tl.uint64}[dtype]

# ------------------------------------------------------------------------------------------------
# Element-wise kernels


@triton.jit
def k_binary(x_ptr, y_ptr, o_ptr, n, OP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    if OP == "add":
        r = x + y
    elif OP == "sub":
        r = x - y
    elif OP == "mul":
        r = x * y
    elif OP == "div":
        r = x / y
    elif OP == "fdiv":
        r = tl.fdiv(x, y)
    elif OP == "div_rn":
        r = tl.div_rn(x, y)
    elif OP == "floordiv":
        r = x // y
    elif OP == "mod":
        r = x % y
    elif OP == "and":
        r = x & y
    elif OP == "or":
        r = x | y
    elif OP == "xor":
        r = x ^ y
    elif OP == "shl":
        r = x << y
    elif OP == "shr":
        r = x >> y
    elif OP == "min":
        r = tl.minimum(x, y)
    elif OP == "max":
        r = tl.maximum(x, y)
    elif OP == "minnan":
        r = tl.minimum(x, y, propagate_nan=tl.PropagateNan.ALL)
    elif OP == "maxnan":
        r = tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL)
    elif OP == "lt":
        r = (x < y).to(o_ptr.dtype.element_ty)
    elif OP == "le":
        r = (x <= y).to(o_ptr.dtype.element_ty)
    elif OP == "eq":
        r = (x == y).to(o_ptr.dtype.element_ty)
    elif OP == "ne":
        r = (x != y).to(o_ptr.dtype.element_ty)
    elif OP == "where":
        r = tl.where(x > y, x - y, y + x)
    elif OP == "umulhi":
        r = tl.umulhi(x, y)
    elif OP == "mix":
        # a few operations in one, so that the compiler schedules them together
        r = (x * y + x) - (y * 3) + (x * x)
    else:
        tl.static_assert(False, "unknown op")
    tl.store(o_ptr + offs, r, mask=m)


@triton.jit
def k_ternary(x_ptr, y_ptr, z_ptr, o_ptr, n, OP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    z = tl.load(z_ptr + offs, mask=m)
    if OP == "fma":
        r = tl.fma(x, y, z)
    elif OP == "clamp":
        r = tl.clamp(x, tl.minimum(y, z), tl.maximum(y, z))
    elif OP == "madd":
        r = x * y + z
    elif OP == "select":
        r = tl.where(z > 0, x, y)
    else:
        tl.static_assert(False, "unknown op")
    tl.store(o_ptr + offs, r, mask=m)


@triton.jit
def k_unary(x_ptr, o_ptr, n, OP: tl.constexpr, BLOCK: tl.constexpr, UP: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    if UP:
        # Triton's functions take float32 and float64 only
        x = x.to(tl.float32)
    if OP == "neg":
        r = -x
    elif OP == "abs":
        r = tl.abs(x)
    elif OP == "not":
        r = ~x
    elif OP == "sqrt":
        r = tl.sqrt(x)
    elif OP == "sqrt_rn":
        r = tl.sqrt_rn(x)
    elif OP == "rsqrt":
        r = tl.rsqrt(x)
    elif OP == "exp":
        r = tl.exp(x)
    elif OP == "exp2":
        r = tl.exp2(x)
    elif OP == "log":
        r = tl.log(x)
    elif OP == "log2":
        r = tl.log2(x)
    elif OP == "sin":
        r = tl.sin(x)
    elif OP == "cos":
        r = tl.cos(x)
    elif OP == "floor":
        r = tl.floor(x)
    elif OP == "ceil":
        r = tl.ceil(x)
    elif OP == "sigmoid":
        r = tl.sigmoid(x)
    elif OP == "erf":
        r = tl.erf(x)
    elif OP == "recip":
        r = 1.0 / x
    elif OP == "square":
        r = x * x
    elif OP == "softplus":
        r = tl.log(1.0 + tl.exp(x))
    else:
        tl.static_assert(False, "unknown op")
    tl.store(o_ptr + offs, r, mask=m)


@triton.jit
def k_convert(x_ptr, o_ptr, n, MODE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    if MODE == "plain":
        r = x.to(o_ptr.dtype.element_ty)
    elif MODE == "rtz":
        r = x.to(o_ptr.dtype.element_ty, fp_downcast_rounding="rtz")
    elif MODE == "bitcast":
        r = x.to(o_ptr.dtype.element_ty, bitcast=True)
    elif MODE == "via32":
        r = x.to(tl.float32).to(o_ptr.dtype.element_ty)
    elif MODE == "viaf16":
        r = x.to(tl.float16).to(o_ptr.dtype.element_ty)
    elif MODE == "viabf16":
        r = x.to(tl.bfloat16).to(o_ptr.dtype.element_ty)
    elif MODE == "viae5":
        r = x.to(tl.float8e5).to(o_ptr.dtype.element_ty)
    elif MODE == "viae4b15":
        r = x.to(tl.float8e4b15).to(o_ptr.dtype.element_ty)
    else:
        tl.static_assert(False, "unknown mode")
    tl.store(o_ptr + offs, r, mask=m)


def ew_make(dtypes_kinds, n, out_dtype, pad=37):
    def make(g):
        t = {}
        for name, (dt, kind) in dtypes_kinds.items():
            t[name] = gen(dt, n + 1, kind, g)
        t['o'] = sentinel(out_dtype, n + pad)
        return t
    return make


def binary_cases():
    out = []
    int_ops = ['add', 'sub', 'mul', 'floordiv', 'mod', 'and', 'or', 'xor', 'shl', 'shr', 'min', 'max',
               'lt', 'eq', 'where', 'mix']
    float_ops = ['add', 'sub', 'mul', 'div', 'fdiv', 'div_rn', 'min', 'max', 'minnan', 'maxnan', 'lt', 'le',
                 'eq', 'ne', 'where', 'mix']
    for dt in ALL:
        ops = float_ops if dt.is_floating_point else int_ops
        if dt in (torch.int32, torch.uint32, torch.int64, torch.uint64):
            ops = ops + ['umulhi']
        for op in ops:
            for kind in (['int', 'rnd', 'special', 'subn'] if dt.is_floating_point else ['int', 'full']):
                if op == 'mix' and dt.is_floating_point and kind != 'int':
                    continue
                if kind == 'subn' and op not in ('add', 'mul', 'div', 'div_rn', 'min', 'max', 'lt', 'where'):
                    continue
                if op in ('floordiv', 'mod'):
                    xk, yk = (kind, 'nz')
                elif op in ('shl', 'shr'):
                    xk, yk = (kind, 'shift')
                elif op == 'mix' and not dt.is_floating_point and kind == 'full':
                    xk, yk = ('mid', 'mid')
                else:
                    xk, yk = kind, kind
                if dt.is_floating_point and kind != 'int' and op in ('div', 'fdiv', 'div_rn', 'mix'):
                    pass
                odt = torch.int8 if op in ('lt', 'le', 'eq', 'ne') else dt
                n = 1 << 16 if kind != 'int' else (1 << 16) - 3
                # division of small integers is exact in no float type: compared within an ulp
                cmp = 'exact'
                ref = 'interp32' if dt == torch.bfloat16 else 'interp'
                if dt.is_floating_point:
                    if op in ('fdiv',) or (op == 'div' and dt == torch.float32):
                        # Triton's float32 division is div.full.f32 (fdiv: div.approx), within 2 ulp
                        # of the quotient, and less exact for divisors beyond 2**126 and subnormal
                        # quotients: those (kind special) between the routes only
                        cmp = ('tol', 2 ** -21, 0)
                        if kind == 'special':
                            ref = None
                if dt == torch.bfloat16 and op in ('div', 'fdiv', 'div_rn', 'mix') and kind != 'int':
                    cmp = ('tol', 2 ** -7, 0) if op != 'mix' else ('tol', 2 ** -5, 1e-30)
                if op in ('umulhi',):
                    ref = umulhi_ref
                if op in ('min', 'max', 'minnan', 'maxnan', 'clamp') and dt.is_floating_point:
                    cmp = 'exact0'  # which zero the minimum of +0 and -0 is, the languages leave open
                if kind == 'subn':
                    ref = None  # Triton's instructions flush subnormals where numpy does not: the routes only
                quick = (op in ('mix', 'floordiv', 'shl', 'umulhi', 'minnan', 'div') and kind in ('full', 'special')
                         and dt in (torch.int64, torch.uint32, torch.int8, torch.float16, torch.bfloat16, torch.float64))
                for block, nw, off in ([(1024, 4, 0), (128, 1, 1)] if kind != 'int' else [(256, 8, 0)]):
                    if dt in (torch.uint16, torch.uint8, torch.int16) and off:
                        pass
                    xa = ('view', 'x', off) if off else 'x'
                    ya = ('view', 'y', off) if off else 'y'
                    r = (lambda t, off=off: umulhi_ref(t, off)) if ref is umulhi_ref else ref
                    out.append(Case(f'bin-{op}-{SHORT[dt]}-{kind}-b{block}w{nw}o{off}', k_binary,
                                    (triton.cdiv(n, block),), [xa, ya, 'o', n],
                                    ew_make({'x': (dt, xk), 'y': (dt, yk)}, n, odt),
                                    ['o'], dict(OP=op, BLOCK=block, num_warps=nw), cmp=cmp, ref=r,
                                    quick=quick and off == 0))
    return out


def umulhi_ref(t, off=0):
    """The upper half of the unsigned product of the two words (`mul.hi.u32`/`.u64`, as Triton
    lowers `umulhi` for signed and unsigned types alike), in Python integers; the inputs from
    element `off` on."""
    x, y, o = t['x'][off:], t['y'][off:], t['o'].clone()
    w = bits(x.dtype)
    mask = (1 << w) - 1
    xs = (x.view(torch.int64) if w == 64 else x.to(torch.int64)).tolist()
    ys = (y.view(torch.int64) if w == 64 else y.to(torch.int64)).tolist()
    hi = [((p & mask) * (q & mask)) >> w for p, q in zip(xs, ys)]
    signed = x.dtype in SINTS or w == 64
    hi = [v - (1 << w) if signed and v >= (1 << (w - 1)) else v for v in hi]
    r = torch.tensor(hi, dtype=torch.int64)
    r = r.view(torch.uint64) if x.dtype == torch.uint64 else r.to(x.dtype)
    m = t['x'].numel() - 1
    o[:m] = r[:m]
    return {'o': o}


def fma_ref(t):
    x, y, z, o = t['x'], t['y'], t['z'], t['o'].clone()
    m = x.numel() - 1
    r = (x.double() * y.double() + z.double()).to(x.dtype)
    o[:m] = r[:m]
    return {'o': o}


def ternary_cases():
    out = []
    for dt in FLOATS:
        for op in ['fma', 'clamp', 'madd', 'select']:
            for kind in ['int', 'rnd', 'special']:
                n = 1 << 15
                ref = 'interp32' if dt == torch.bfloat16 else 'interp'
                cmp = 'exact'
                if kind != 'int' and op in ('fma', 'madd'):
                    # the interpreter rounds the product, the GPU fuses it: the product and the sum in
                    # float64 (exact for float16 and float32 but for a double rounding), within an ulp
                    ref = fma_ref if dt != torch.float64 else None
                    cmp = ('tol', torch.finfo(dt).eps, torch.finfo(dt).tiny)
                if op == 'clamp':
                    cmp = 'exact0'  # which zero, the languages leave open
                if op == 'fma' and dt == torch.bfloat16:
                    continue
                out.append(Case(f'ter-{op}-{SHORT[dt]}-{kind}', k_ternary, (triton.cdiv(n, 512),),
                                ['x', 'y', 'z', 'o', n], ew_make({'x': (dt, kind), 'y': (dt, kind), 'z': (dt, kind)}, n, dt),
                                ['o'], dict(OP=op, BLOCK=512, num_warps=4), cmp=cmp, ref=ref,
                                quick=(op == 'fma' and kind == 'special' and dt in (torch.float32, torch.float64))))
    for dt in INTS:
        for op in ['clamp', 'madd', 'select']:
            n = 1 << 15
            kind = 'full' if op != 'madd' else 'mid'
            out.append(Case(f'ter-{op}-{SHORT[dt]}-{kind}', k_ternary, (triton.cdiv(n, 512),),
                            ['x', 'y', 'z', 'o', n], ew_make({'x': (dt, kind), 'y': (dt, kind), 'z': (dt, kind)}, n, dt),
                            ['o'], dict(OP=op, BLOCK=512, num_warps=4)))
    return out


UNARY_TOL = {
    # (rtol, atol) against the interpreter (numpy), for the approximate instructions Triton uses
    'sqrt': (2 ** -22, 0), 'rsqrt': (2 ** -21, 0), 'exp': (2 ** -20, 1e-37), 'exp2': (2 ** -21, 1e-37),
    'log': (2 ** -20, 2 ** -21), 'log2': (2 ** -21, 2 ** -21), 'sin': (2 ** -18, 2 ** -20), 'cos': (2 ** -18, 2 ** -20),
    'sigmoid': (2 ** -20, 1e-37), 'erf': (2 ** -20, 2 ** -22), 'recip': (2 ** -22, 0), 'softplus': (2 ** -19, 2 ** -20),
}


def unary_cases():
    out = []
    for dt in ALL:
        if dt.is_floating_point:
            ops = ['neg', 'abs', 'sqrt', 'sqrt_rn', 'rsqrt', 'exp', 'exp2', 'log', 'log2', 'sin', 'cos', 'floor',
                   'ceil', 'sigmoid', 'erf', 'recip', 'square', 'softplus']
            kinds = ['rnd', 'special', 'wide', 'subn']
        else:
            ops = ['neg', 'abs', 'not', 'square']
            kinds = ['full']
        for op in ops:
            for kind in kinds:
                if op in ('sqrt', 'rsqrt', 'log', 'log2', 'sqrt_rn') and kind == 'rnd':
                    xk = 'unit'
                else:
                    xk = kind
                n = (1 << 16) - 5
                ref = 'interp32' if dt == torch.bfloat16 else 'interp'
                cmp = 'exact'
                if op in UNARY_TOL:
                    r, a = UNARY_TOL[op]
                    if dt in (torch.float16, torch.bfloat16):
                        r, a = max(r, torch.finfo(dt).eps), max(a, 2 ** -24 if dt == torch.float16 else 1e-37)
                    if dt == torch.float64 and op not in ('sqrt', 'recip'):
                        # Triton's float64 functions are libdevice's (exact to an ulp or two)
                        r, a = (2 ** -50 if op != 'softplus' else 2 ** -47, 1e-300)
                    if dt == torch.float64 and op in ('sqrt', 'recip'):
                        r, a = (0, 0)
                    cmp = ('tol', r, a) if (r or a) else 'exact'
                if op == 'sqrt_rn':
                    cmp = 'exact'
                if op in ('sin', 'cos') and kind != 'rnd':
                    # the GPU's approximation is not meant for large arguments
                    continue
                if op in ('exp', 'exp2', 'sigmoid', 'softplus', 'erf') and kind == 'wide':
                    continue
                if kind == 'subn':
                    ref = None  # Triton's approximations and roundings flush subnormals (.ftz): the routes only
                quick = (op in ('exp', 'rsqrt', 'sqrt_rn', 'floor', 'neg') and kind == 'special'
                         and dt in (torch.float32, torch.bfloat16, torch.float16)) or (op == 'neg' and dt == torch.int64)
                up = dt in (torch.float16, torch.bfloat16) and op not in ('neg', 'abs', 'recip', 'square')
                out.append(Case(f'un-{op}-{SHORT[dt]}-{kind}', k_unary, (triton.cdiv(n, 512),), ['x', 'o', n],
                                ew_make({'x': (dt, xk)}, n, dt), ['o'], dict(OP=op, BLOCK=512, num_warps=4, UP=up),
                                cmp=cmp, ref=ref, quick=quick))
    return out


def convert_ref(mode):
    """What a conversion gives, computed by PyTorch: the interpreter's bfloat16 conversion
    truncates, the GPU's (and PyTorch's) rounds to nearest even."""
    def ref(t):
        x, o = t['x'], t['o'].clone()
        m = x.numel() - 1
        src = x[:m]
        if mode == 'plain':
            r = src.to(o.dtype)
        elif mode == 'bitcast':
            r = src.view(o.dtype)
        elif mode == 'via32':
            r = src.to(torch.float32).to(o.dtype)
        elif mode == 'viaf16':
            r = src.to(torch.float16).to(o.dtype)
        elif mode == 'viabf16':
            r = src.to(torch.bfloat16).to(o.dtype)
        else:
            raise ValueError(mode)
        o[:m] = r
        return {'o': o}
    return ref


def conv_kind(src, dst):
    """Data of src that converts to dst without leaving dst's range (no conversion whose result
    the languages leave undefined)."""
    def make_fn(g, n):
        if src.is_floating_point:
            v = gen(src, n, 'rnd', g).double() * 40
            if not dst.is_floating_point:
                info = torch.iinfo(dst)
                lo, hi = max(info.min, -2.0 ** 60), min(info.max, 2.0 ** 60)
                scale = torch.pow(2.0, torch.randint(0, 64, (n,), generator=g).double())
                v = torch.randn(n, generator=g, dtype=torch.float64) * scale
                v = v.clamp(lo / 2, hi / 2)
                if src == torch.float16:
                    v = v.clamp(-60000, 60000)
                if src == torch.bfloat16:
                    pass
                return v.to(src)
            return gen(src, n, 'special', g)
        info = torch.iinfo(src)
        v = gen(src, n, 'full', g)
        if dst == torch.float16:
            # beyond 65504 the result is infinite on the GPU and in numpy alike, but keep both kinds
            pass
        return v
    return make_fn


def convert_cases():
    out = []
    for src in ALL:
        for dst in ALL:
            if src == dst:
                continue
            for mode in ['plain', 'rtz', 'bitcast']:
                if mode == 'bitcast' and bits(src) != bits(dst):
                    continue
                if mode == 'rtz' and not (src.is_floating_point and dst.is_floating_point and bits(dst) < bits(src)):
                    continue
                if mode == 'rtz' and src == torch.float16 and dst == torch.bfloat16:
                    continue
                n = 1 << 14
                variants = []
                if mode == 'rtz':
                    # the interpreter makes NaN of what overflows: in-range data against it (not for
                    # bfloat16, which it truncates), the rest (infinities, overflow, NaN, subnormals)
                    # between the routes only
                    variants.append(('', lambda g, n, s=src: gen(s, n, 'rnd', g), 'interp' if torch.bfloat16 not in (src, dst) else None))
                    variants.append(('-subn', lambda g, n, s=src: gen(s, n, 'subn', g), None))
                elif mode == 'bitcast':
                    variants.append(('', lambda g, n, s=src: gen(s, n, 'full', g) if not s.is_floating_point else gen(s, n, 'subn', g),
                                     convert_ref(mode) if torch.bfloat16 in (src, dst) else 'interp'))
                else:
                    variants.append(('', conv_kind(src, dst), convert_ref(mode) if torch.bfloat16 in (src, dst) else 'interp'))
                    if src.is_floating_point and dst.is_floating_point:
                        variants.append(('-subn', lambda g, n, s=src: gen(s, n, 'subn', g), None))
                quick = (src, dst, mode) in ((torch.float32, torch.bfloat16, 'plain'), (torch.int64, torch.float32, 'plain'),
                                             (torch.float64, torch.uint64, 'plain'), (torch.float32, torch.float16, 'rtz'),
                                             (torch.uint64, torch.float16, 'plain'), (torch.float64, torch.bfloat16, 'plain'))
                for suffix, fn, ref in variants:
                    def make(g, fn=fn, n=n, dst=dst):
                        return {'x': fn(g, n + 1), 'o': sentinel(dst, n + 21)}
                    out.append(Case(f'cvt-{mode}-{SHORT[src]}-{SHORT[dst]}{suffix}', k_convert, (triton.cdiv(n, 256),),
                                    ['x', 'o', n], make, ['o'], dict(MODE=mode, BLOCK=256, num_warps=2), ref=ref,
                                    quick=quick and not suffix))
    # through the narrow types, from every float type
    for src in FLOATS:
        for mode in ['via32', 'viaf16', 'viabf16', 'viae5', 'viae4b15']:
            n = 1 << 14

            def make(g, src=src, n=n):
                return {'x': gen(src, n + 1, 'special', g), 'o': sentinel(torch.float32, n + 21)}
            ref = convert_ref(mode) if mode in ('via32', 'viaf16', 'viabf16') else None
            # (Through the 8-bit floats, Triton's conversion for sm_75 is its own sequence: halves of
            # an ulp rounded away from zero, NaN of float32 made -0, infinities of bfloat16 made NaN.
            # The interpreter does otherwise. The routes only.)
            out.append(Case(f'cvt-{mode}-{SHORT[src]}', k_convert, (triton.cdiv(n, 256),), ['x', 'o', n], make, ['o'],
                            dict(MODE=mode, BLOCK=256, num_warps=2), ref=ref,
                            quick=mode in ('viae5', 'viabf16') and src == torch.float32))
    return out

# ------------------------------------------------------------------------------------------------
# Reductions, scans, sorts


@triton.jit
def k_reduce(x_ptr, o_ptr, i_ptr, M, N, stride, OP: tl.constexpr, AXIS: tl.constexpr, OTHER: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    m = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(x_ptr + rows[:, None] * stride + cols[None, :], mask=m, other=OTHER)
    if AXIS == 1:
        if OP == "sum":
            r = tl.sum(x, axis=1)
        elif OP == "max":
            r = tl.max(x, axis=1)
        elif OP == "min":
            r = tl.min(x, axis=1)
        elif OP == "xor":
            r = tl.xor_sum(x, axis=1)
        elif OP == "or":
            r = tl.reduce_or(x, axis=1)
        elif OP == "argmax":
            r = tl.argmax(x, axis=1)
        elif OP == "argmin":
            r = tl.argmin(x, axis=1)
        elif OP == "maxidx":
            v, j = tl.max(x, axis=1, return_indices=True)
            r = v
            tl.store(i_ptr + rows, j, mask=rows < M)
        elif OP == "sumsq":
            r = tl.sum(x * x, axis=1)
        else:
            tl.static_assert(False, "op")
        if OP == "argmax" or OP == "argmin":
            tl.store(i_ptr + rows, r, mask=rows < M)
        else:
            tl.store(o_ptr + rows, r, mask=rows < M)
    elif AXIS == 0:
        if OP == "sum":
            r = tl.sum(x, axis=0)
        elif OP == "max":
            r = tl.max(x, axis=0)
        elif OP == "min":
            r = tl.min(x, axis=0)
        elif OP == "xor":
            r = tl.xor_sum(x, axis=0)
        elif OP == "or":
            r = tl.reduce_or(x, axis=0)
        elif OP == "argmax":
            r = tl.argmax(x, axis=0)
        elif OP == "argmin":
            r = tl.argmin(x, axis=0)
        elif OP == "maxidx":
            v, j = tl.max(x, axis=0, return_indices=True)
            r = v
            tl.store(i_ptr + pid * BN + cols, j, mask=cols < N)
        elif OP == "sumsq":
            r = tl.sum(x * x, axis=0)
        else:
            tl.static_assert(False, "op")
        if OP == "argmax" or OP == "argmin":
            tl.store(i_ptr + pid * BN + cols, r, mask=cols < N)
        else:
            tl.store(o_ptr + pid * BN + cols, r, mask=cols < N)
    else:
        if OP == "sum":
            r = tl.sum(tl.sum(x, axis=1), axis=0)
        elif OP == "max":
            r = tl.max(tl.reshape(x, (BM * BN,)), axis=0)
        elif OP == "min":
            r = tl.min(tl.reshape(x, (BM * BN,)), axis=0)
        elif OP == "xor":
            r = tl.xor_sum(tl.reshape(x, (BM * BN,)), axis=0)
        elif OP == "or":
            r = tl.reduce_or(tl.reshape(x, (BM * BN,)), axis=0)
        elif OP == "argmax":
            r = tl.argmax(tl.reshape(x, (BM * BN,)), axis=0)
        elif OP == "argmin":
            r = tl.argmin(tl.reshape(x, (BM * BN,)), axis=0)
        elif OP == "sumsq":
            r = tl.sum(tl.reshape(x * x, (BM * BN,)), axis=0)
        else:
            tl.static_assert(False, "op")
        if OP == "argmax" or OP == "argmin":
            tl.store(i_ptr + pid, r)
        else:
            tl.store(o_ptr + pid, r)


@triton.jit
def _combine_mulmod(a, b):
    return (a * b) & 0xFFFF


@triton.jit
def _combine_pair_max(a, ia, b, ib):
    take = (b > a) | ((b == a) & (ib < ia))
    return tl.where(take, b, a), tl.where(take, ib, ia)


@triton.jit
def k_reduce_custom(x_ptr, o_ptr, i_ptr, N, OP: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BN)
    m = cols < N
    if OP == "mulmod":
        x = tl.load(x_ptr + pid * N + cols, mask=m, other=1)
        tl.store(o_ptr + pid, tl.reduce(x, 0, _combine_mulmod))
    else:
        x = tl.load(x_ptr + pid * N + cols, mask=m, other=-1000000)
        v, j = tl.reduce((x, cols), 0, _combine_pair_max)
        tl.store(o_ptr + pid, v)
        tl.store(i_ptr + pid, j)


@triton.jit
def k_row_loop(x_ptr, o_ptr, M, N, stride, OP: tl.constexpr, BN: tl.constexpr):
    """A row of any length, in chunks: the shape of softmax, norms and sums over a long axis."""
    row = tl.program_id(0)
    base = x_ptr + row.to(tl.int64) * stride
    if OP == "sum":
        acc = tl.zeros((BN,), dtype=tl.float32)
        for start in range(0, N, BN):
            cols = start + tl.arange(0, BN)
            acc += tl.load(base + cols, mask=cols < N, other=0.0).to(tl.float32)
        tl.store(o_ptr + row, tl.sum(acc, axis=0))
    elif OP == "softmax":
        mx = tl.full((BN,), float("-inf"), tl.float32)
        for start in range(0, N, BN):
            cols = start + tl.arange(0, BN)
            mx = tl.maximum(mx, tl.load(base + cols, mask=cols < N, other=float("-inf")).to(tl.float32))
        rmax = tl.max(mx, axis=0)
        den = tl.zeros((BN,), dtype=tl.float32)
        for start in range(0, N, BN):
            cols = start + tl.arange(0, BN)
            v = tl.load(base + cols, mask=cols < N, other=float("-inf")).to(tl.float32)
            den += tl.exp(v - rmax)
        d = tl.sum(den, axis=0)
        for start in range(0, N, BN):
            cols = start + tl.arange(0, BN)
            v = tl.load(base + cols, mask=cols < N, other=float("-inf")).to(tl.float32)
            tl.store(o_ptr + row.to(tl.int64) * stride + cols, (tl.exp(v - rmax) / d).to(o_ptr.dtype.element_ty),
                     mask=cols < N)
    elif OP == "rmsnorm":
        acc = tl.zeros((BN,), dtype=tl.float32)
        for start in range(0, N, BN):
            cols = start + tl.arange(0, BN)
            v = tl.load(base + cols, mask=cols < N, other=0.0).to(tl.float32)
            acc += v * v
        r = tl.rsqrt(tl.sum(acc, axis=0) / N + 1e-6)
        for start in range(0, N, BN):
            cols = start + tl.arange(0, BN)
            v = tl.load(base + cols, mask=cols < N, other=0.0).to(tl.float32)
            tl.store(o_ptr + row.to(tl.int64) * stride + cols, (v * r).to(o_ptr.dtype.element_ty), mask=cols < N)
    else:
        tl.static_assert(False, "op")


def reduce_ref_other(op, dt):
    if op in ('sum', 'xor', 'or', 'sumsq'):
        return 0
    big = op in ('min', 'argmin')
    if dt.is_floating_point:
        return float('inf') if big else float('-inf')
    info = torch.iinfo(dt)
    return info.max if big else info.min


def reduce_cases():
    out = []
    shapes = [(1, 1024, 4), (32, 32, 4), (4, 512, 8), (128, 16, 4), (64, 64, 1), (16, 256, 16), (2, 4096, 8), (256, 8, 2)]
    for dt in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint32, torch.uint8, torch.float16, torch.bfloat16,
               torch.float32, torch.float64]:
        ops = ['sum', 'max', 'min', 'argmax', 'argmin', 'maxidx', 'sumsq']
        if not dt.is_floating_point:
            ops += ['xor', 'or']
        for op in ops:
            for axis in [1, 0, -1]:
                if op == 'maxidx' and axis == -1:
                    continue
                for si, (bm, bn, nw) in enumerate(shapes):
                    if (si + len(op) + axis) % 3 != 0 and not (op == 'sum' and dt == torch.float32):
                        continue
                    M, N = bm * 37 + 3, bn - (1 if bn > 8 else 0)
                    grid = (triton.cdiv(M, bm),)
                    kind = 'int' if dt.is_floating_point else 'full'
                    if op in ('sum', 'sumsq') and not dt.is_floating_point:
                        kind = 'small'
                    if dt.is_floating_point and op in ('sum', 'sumsq'):
                        kind = 'small' if dt in (torch.float16, torch.bfloat16) else 'int'
                    if op in ('argmax', 'argmin', 'maxidx', 'max', 'min') and dt.is_floating_point:
                        kind = 'rnd'
                    if axis == 1:
                        osz, isz = M + 9, M + 9
                    elif axis == 0:
                        osz, isz = grid[0] * bn + 9, grid[0] * bn + 9
                    else:
                        osz, isz = grid[0] + 9, grid[0] + 9
                    odt = dt
                    if op in ('sum', 'sumsq') and dt in (torch.int8, torch.int16, torch.uint8):
                        odt = torch.int32
                    if op in ('sum', 'sumsq') and dt in (torch.float16, torch.bfloat16):
                        odt = torch.float32

                    def make(g, dt=dt, M=M, N=N, kind=kind, osz=osz, isz=isz, odt=odt):
                        return {'x': gen(dt, (M, N), kind, g), 'o': sentinel(odt, osz), 'i': sentinel(torch.int32, isz)}
                    ref = 'interp32' if dt == torch.bfloat16 else 'interp'
                    cmp = 'exact'
                    if op in ('sum', 'sumsq') and dt in (torch.float16, torch.bfloat16):
                        # Triton gives the sum of 16-bit floats in their type, the interpreter (and the
                        # float32 copy) in a wider one: within an ulp of the 16-bit type
                        cmp = ('tol', 2 * torch.finfo(dt).eps, 0)
                    other = reduce_ref_other(op if op != 'maxidx' else 'max', dt)
                    quick = (op in ('argmax', 'sum', 'xor') and dt in (torch.float32, torch.int64, torch.bfloat16)
                             and si in (1, 6))
                    out.append(Case(f'red-{op}-{SHORT[dt]}-ax{axis}-{bm}x{bn}w{nw}', k_reduce, grid,
                                    ['x', 'o', 'i', M, N, N], make, ['o', 'i'],
                                    dict(OP=op, AXIS=axis, OTHER=other, BM=bm, BN=bn, num_warps=nw), cmp=cmp, ref=ref,
                                    quick=quick))
    for op in ['mulmod', 'pairmax']:
        for bn, nw in [(32, 1), (256, 4), (1024, 8), (2048, 16)]:
            M, N = 300, bn - 3

            def make(g, M=M, N=N, op=op):
                if op == 'mulmod':
                    x = gen(torch.int32, (M, N), 'pos', g)
                else:
                    x = gen(torch.int32, (M, N), 'int', g)
                return {'x': x, 'o': sentinel(torch.int32, M + 5), 'i': sentinel(torch.int32, M + 5)}
            out.append(Case(f'redc-{op}-{bn}w{nw}', k_reduce_custom, (M,), ['x', 'o', 'i', N], make, ['o', 'i'],
                            dict(OP=op, BN=bn, num_warps=nw), quick=bn == 256))
    for op in ['sum', 'softmax', 'rmsnorm']:
        for dt in [torch.float32, torch.float16, torch.bfloat16]:
            for (M, N, bn, nw) in [(64, 5000, 1024, 4), (513, 128, 128, 1), (8, 70000, 4096, 16), (300, 1000, 256, 8)]:
                kind = 'int' if op == 'sum' else 'rnd'
                odt = torch.float32 if op == 'sum' else dt
                osz = M + 3 if op == 'sum' else M * N + 3

                def make(g, M=M, N=N, dt=dt, kind=kind, odt=odt, osz=osz):
                    return {'x': gen(dt, (M, N), kind, g), 'o': sentinel(odt, osz)}
                cmp = 'exact' if op == 'sum' else ('tol', 2 ** -8 if dt != torch.float32 else 2 ** -19, 1e-6 if op == 'softmax' else 2 ** -10)
                if dt == torch.bfloat16 and op != 'sum':
                    cmp = ('tol', 2 ** -6, 1e-4)
                ref = 'interp32' if dt == torch.bfloat16 else 'interp'
                out.append(Case(f'rowloop-{op}-{SHORT[dt]}-{M}x{N}b{bn}w{nw}', k_row_loop, (M,), ['x', 'o', M, N, N], make,
                                ['o'], dict(OP=op, BN=bn, num_warps=nw), cmp=cmp, ref=ref,
                                quick=(N == 5000 and dt != torch.float16) or (N == 70000 and op == 'softmax' and dt == torch.float32)))
    return out


@triton.jit
def _combine_max(a, b):
    return tl.maximum(a, b)


@triton.jit
def _combine_affine(a1, b1, a2, b2):
    # composition of x -> a*x + b (the scan of a linear recurrence, as in state-space models)
    return a1 * a2, b1 * a2 + b2


@triton.jit
def k_scan(x_ptr, y_ptr, o_ptr, p_ptr, M, N, OP: tl.constexpr, AXIS: tl.constexpr, REV: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    m = (rows[:, None] < M) & (cols[None, :] < N)
    offs = rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptr + offs, mask=m, other=0)
    if OP == "cumsum":
        r = tl.cumsum(x, axis=AXIS, reverse=REV)
    elif OP == "cumprod":
        r = tl.cumprod(x, axis=AXIS, reverse=REV)
    elif OP == "cummax":
        r = tl.associative_scan(x, AXIS, _combine_max, reverse=REV)
    elif OP == "affine":
        y = tl.load(y_ptr + offs, mask=m, other=0)
        a, r = tl.associative_scan((x, y), AXIS, _combine_affine, reverse=REV)
        tl.store(p_ptr + offs, a, mask=m)
    else:
        tl.static_assert(False, "op")
    tl.store(o_ptr + offs, r, mask=m)


def scan_cases():
    out = []
    for dt in [torch.int32, torch.int64, torch.float32, torch.float16, torch.float64, torch.int8, torch.uint32, torch.bfloat16]:
        for op in ['cumsum', 'cumprod', 'cummax', 'affine']:
            if op == 'affine' and not dt.is_floating_point and dt != torch.int32:
                continue
            for axis in [1, 0]:
                for rev in [False, True]:
                    for (bm, bn, nw) in [(1, 1024, 4), (16, 64, 4), (64, 32, 8), (4, 256, 1), (128, 128, 8)]:
                        if (bm + bn + int(rev) + axis + len(op)) % 2 and not (dt == torch.float32 and op == 'cumsum'):
                            continue
                        M, N = bm * 9 + 1, bn - 1
                        if op == 'cumprod':
                            kind = 'small'
                        elif op == 'cummax':
                            kind = 'full' if not dt.is_floating_point else 'rnd'
                        else:
                            kind = 'small' if dt in (torch.float16, torch.bfloat16, torch.int8) else 'int'
                        if op == 'affine':
                            kind = 'small'
                        osz = M * N + 7

                        def make(g, dt=dt, M=M, N=N, kind=kind, osz=osz, op=op):
                            x = gen(dt, (M, N), kind, g)
                            if op == 'cumprod' or op == 'affine':
                                # products of many small integers overflow: mostly ones and minus ones, few twos
                                sel = torch.randint(0, 16, (M, N), generator=g)
                                x = torch.where(sel == 0, gen(dt, (M, N), 'small', g), torch.ones((M, N), dtype=dt))
                                if not (dt in UINTS):
                                    x = torch.where(sel == 1, -x, x)
                            return {'x': x, 'y': gen(dt, (M, N), 'small', g), 'o': sentinel(dt, osz), 'p': sentinel(dt, osz)}
                        cmp = 'exact'
                        if op == 'cumprod' and dt in (torch.float16, torch.bfloat16):
                            pass
                        ref = 'interp32' if dt == torch.bfloat16 else 'interp'
                        if op == 'affine' and dt in (torch.float16, torch.bfloat16):
                            ref = None  # rounded at each step in 16 bits on the GPU, not by the interpreter
                        quick = (op in ('cumsum', 'affine') and dt in (torch.float32, torch.int64) and bm == 16)
                        out.append(Case(f'scan-{op}-{SHORT[dt]}-ax{axis}-r{int(rev)}-{bm}x{bn}w{nw}', k_scan,
                                        (triton.cdiv(M, bm),), ['x', 'y', 'o', 'p', M, N], make, ['o', 'p'],
                                        dict(OP=op, AXIS=axis, REV=rev, BM=bm, BN=bn, num_warps=nw), cmp=cmp, ref=ref,
                                        quick=quick))
    return out


@triton.jit
def k_sort(x_ptr, o_ptr, i_ptr, N, OP: tl.constexpr, DESC: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
           BN: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    offs = rows[:, None] * BN + cols[None, :]
    x = tl.load(x_ptr + offs)
    if OP == "sort":
        tl.store(o_ptr + offs, tl.sort(x, dim=1, descending=DESC))
    elif OP == "topk":
        kc = tl.arange(0, K)
        tl.store(o_ptr + rows[:, None] * K + kc[None, :], tl.topk(x, K, dim=1))
    elif OP == "flip":
        tl.store(o_ptr + offs, tl.flip(x, dim=1))
    elif OP == "gather":
        idx = tl.load(i_ptr + offs)
        tl.store(o_ptr + offs, tl.gather(x, idx, 1))
    elif OP == "hist":
        h = tl.histogram(tl.reshape(x, (BM * BN,)).to(tl.int32) & (K - 1), K)
        tl.store(i_ptr + pid * K + tl.arange(0, K), h)
    elif OP == "trans":
        xt = tl.trans(x)
        offs_t = pid * BM * BN + tl.arange(0, BN)[:, None] * BM + tl.arange(0, BM)[None, :]
        tl.store(o_ptr + offs_t, xt)
    elif OP == "interleave":
        a, b = tl.split(tl.reshape(x, (BM, BN // 2, 2)))
        y = tl.interleave(b, a)
        tl.store(o_ptr + offs, y)
    elif OP == "join":
        j = tl.join(x, x + 1)
        tl.store(o_ptr + rows[:, None, None] * (BN * 2) + cols[None, :, None] * 2 + tl.arange(0, 2)[None, None, :], j)
    elif OP == "cat":
        y = tl.cat(x, x * 2, can_reorder=False)
        tl.store(o_ptr + pid * BM * BN * 2 + tl.arange(0, 2 * BM)[:, None] * BN + cols[None, :], y)
    elif OP == "permute":
        y = tl.permute(tl.reshape(x, (BM, 2, BN // 2)), (2, 0, 1))
        r = tl.arange(0, BN // 2)[:, None, None] * (BM * 2) + tl.arange(0, BM)[None, :, None] * 2 + tl.arange(0, 2)[None, None, :]
        tl.store(o_ptr + pid * BM * BN + r, y)
    else:
        tl.static_assert(False, "op")


def sort_cases():
    out = []
    for dt in [torch.float32, torch.int32, torch.float16, torch.int16, torch.int64, torch.uint8, torch.bfloat16, torch.float64]:
        for op in ['sort', 'topk', 'flip', 'gather', 'hist', 'trans', 'interleave', 'join', 'cat', 'permute']:
            for (bm, bn, nw) in [(1, 128, 1), (4, 256, 4), (16, 32, 2), (2, 1024, 8), (32, 16, 4), (1, 2048, 16)]:
                if op == 'topk' and dt in (torch.uint8,):
                    continue
                if (bm + bn + len(op) + bits(dt)) % 3 and not (dt == torch.float32):
                    continue
                if op == 'hist' and dt not in (torch.int32, torch.int16, torch.float32, torch.uint8):
                    continue
                M = bm * 13
                K = 16 if op == 'topk' else 32
                if op == 'topk' and bn < 32:
                    K = 8
                kind = 'rnd' if dt.is_floating_point else 'full'
                if op == 'hist':
                    kind = 'pos' if dt.is_floating_point else 'full'
                osz = {'topk': M * K, 'hist': 0, 'join': M * bn * 2, 'cat': M * bn * 2}.get(op, M * bn) + 11
                isz = M * bn + 11 if op != 'hist' else (M // bm) * K + 11

                def make(g, dt=dt, M=M, bn=bn, kind=kind, osz=osz, isz=isz, op=op):
                    t = {'x': gen(dt, (M, bn), kind, g), 'o': sentinel(dt, osz)}
                    if op == 'gather':
                        t['i'] = torch.randint(0, bn, (M * bn + 11,), generator=g, dtype=torch.int32)
                    else:
                        t['i'] = sentinel(torch.int32, isz)
                    return t
                ref = 'interp32' if dt == torch.bfloat16 else 'interp'
                if op == 'hist' and dt == torch.float32:
                    ref = 'interp'
                # (And the gather of floats in 16 warps: blocks of 512 threads, which its cubin
                # requires, allow 128 registers, and its translation takes 137 at every level. Until
                # 2026-11-23 it was given so, and Triton stopped before the launch; it is refused
                # since, and runs from the PTX that its cubin keeps: PASS-PTX.)
                quick = ((dt == torch.float32 and nw == 4 and bn <= 256)
                         or (op in ('sort', 'topk') and dt in (torch.int64, torch.float16) and nw == 4 and bn <= 256)
                         or (op == 'gather' and dt == torch.float32 and nw == 16))
                for desc in ([False, True] if op == 'sort' else [False]):
                    out.append(Case(f'sort-{op}-{SHORT[dt]}-d{int(desc)}-{bm}x{bn}w{nw}', k_sort, (M // bm,),
                                    ['x', 'o', 'i', bn], make, ['o', 'i'],
                                    dict(OP=op, DESC=desc, K=K, BM=bm, BN=bn, num_warps=nw), ref=ref, quick=quick))
    return out

# ------------------------------------------------------------------------------------------------
# Matrix products


@triton.jit
def k_mm(a_ptr, b_ptr, c_ptr, bias_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, PREC: tl.constexpr,
         ACC: tl.constexpr, EPI: tl.constexpr, MASK_K: tl.constexpr, ACC_IN: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = GROUP_M * num_n
    gid = pid // group
    first_m = gid * GROUP_M
    gsize = min(num_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % group) % gsize)
    pid_n = (pid % group) // gsize
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    b_ptrs = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=ACC)
    for k in range(0, tl.cdiv(K, BK)):
        if MASK_K:
            kk = K - k * BK
            a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < kk), other=0)
            b = tl.load(b_ptrs, mask=(rk[:, None] < kk) & (rn[None, :] < N), other=0)
        else:
            a = tl.load(a_ptrs, mask=rm[:, None] < M, other=0)
            b = tl.load(b_ptrs, mask=rn[None, :] < N, other=0)
        if ACC_IN:
            acc = tl.dot(a, b, acc, input_precision=PREC, out_dtype=ACC)
        else:
            acc += tl.dot(a, b, input_precision=PREC, out_dtype=ACC)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    if EPI == "bias":
        bias = tl.load(bias_ptr + rn, mask=rn < N, other=0)
        acc = acc + bias[None, :].to(ACC)
    elif EPI == "relu":
        acc = tl.where(acc > 0, acc, acc * 0.25)
    c = acc.to(c_ptr.dtype.element_ty)
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, c, mask=(rm[:, None] < M) & (rn[None, :] < N))


def mm_case(name, adt, bdt, cdt, acc, prec, M, N, K, bm, bn, bk, nw, ns, trans_a=False, trans_b=False, epi='none',
            kind='int', group=8, acc_in=True, quick=False, cmp='exact'):
    mask_k = K % bk != 0

    def make(g):
        a = gen(adt, (K, M) if trans_a else (M, K), kind, g)
        b = gen(bdt, (N, K) if trans_b else (K, N), kind, g)
        return {'a': a, 'b': b, 'bias': gen(cdt if cdt != torch.int32 else torch.int32, (N,), 'small', g),
                'c': sentinel(cdt, (M + 1, N))}

    def launch(t, interpreted):
        a, b = t['a'], t['b']
        sam, sak = (1, M) if trans_a else (K, 1)
        sbk, sbn = (1, K) if trans_b else (N, 1)
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
        meta = dict(BM=bm, BN=bn, BK=bk, GROUP_M=group, PREC=prec, ACC=acc, EPI=epi, MASK_K=mask_k, ACC_IN=acc_in)
        if not interpreted:
            meta.update(num_warps=nw, num_stages=ns)
        k_mm[grid](a, b, t['c'], t['bias'], M, N, K, sam, sak, sbk, sbn, N, 1, **meta)
    ref = 'interp32' if torch.bfloat16 in (adt, bdt, cdt) else 'interp'
    return Case(name, k_mm, None, None, make, ['c'], cmp=cmp, ref=ref, launch=launch, quick=quick)


# Cases that the translator of 0.1.2 got wrong (lane numbers stored for accumulators, different in
# every run: the defect fixed in 0.1.3), measured on a V100 on 2026-11-23: the quick test has them.
MM_SEEN_WRONG = {
    'mm-bf16f32-accf32-ieee-33x1000x40-64x256x32w8s2-t11-relu',
    'mm-f16f32-accf32-tf32-33x1000x40-128x128x32w8s3-t10-relu',
    'mm-f16f32-accf32-tf32-33x1000x40-128x128x64w8s3-t00-none',
}


def mm_cases():
    out = []
    tiles = [(16, 16, 16, 1), (32, 32, 16, 2), (64, 64, 32, 4), (64, 64, 16, 4), (128, 64, 32, 4), (64, 128, 32, 8),
             (128, 128, 32, 8), (32, 64, 64, 4), (16, 64, 32, 2), (64, 32, 8, 4), (128, 128, 64, 8), (32, 32, 32, 1),
             (256, 64, 16, 8), (64, 256, 32, 8), (64, 64, 64, 2), (128, 32, 16, 4), (16, 128, 128, 4)]
    kinds = [
        # (a, b, c, acc, prec)
        (torch.float16, torch.float16, torch.float32, tl.float32, 'ieee'),
        (torch.float16, torch.float16, torch.float16, tl.float32, 'ieee'),
        (torch.float16, torch.float16, torch.float16, tl.float16, 'ieee'),
        (torch.bfloat16, torch.bfloat16, torch.float32, tl.float32, 'ieee'),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16, tl.float32, 'ieee'),
        (torch.float32, torch.float32, torch.float32, tl.float32, 'ieee'),
        (torch.float32, torch.float32, torch.float32, tl.float32, 'tf32'),
        (torch.float32, torch.float32, torch.float32, tl.float32, 'tf32x3'),
        (torch.int8, torch.int8, torch.int32, tl.int32, 'ieee'),
        (torch.float64, torch.float64, torch.float64, tl.float64, 'ieee'),
        (torch.float16, torch.float16, torch.float32, tl.float32, 'tf32'),
    ]
    shapes = [(512, 384, 192), (203, 157, 99), (1024, 512, 64), (64, 64, 512), (33, 1000, 40)]
    i = 0
    for ki, (adt, bdt, cdt, acc, prec) in enumerate(kinds):
        for ti, (bm, bn, bk, nw) in enumerate(tiles):
            for si, (M, N, K) in enumerate(shapes):
                i += 1
                if (ki * 7 + ti * 3 + si) % 4 != 0:
                    continue
                if adt == torch.float64 and bm * bn > 64 * 64:
                    continue
                if bk < 16 and adt not in (torch.float32, torch.float64):
                    continue  # tl.dot takes K of 16 or more for these types
                if adt == torch.int8 and ti > 4:
                    continue  # Triton 3.8 cannot compile an int8 product for sm_75: a few cases say so
                ns = [1, 2, 3, 4][(ti + si) % 4]
                ta, tb = bool((ti + ki) % 3 == 1), bool((ti + si) % 2 == 1)
                epi = ['none', 'bias', 'relu'][(ki + si + ti) % 3] if acc != tl.int32 else 'none'
                if epi == 'relu' and cdt != torch.float32 and acc != tl.float64:
                    epi = 'none'
                kind = 'small' if cdt in (torch.float16, torch.bfloat16) or acc == tl.float16 else 'int'
                if cdt in (torch.float16, torch.bfloat16) and K > 128:
                    K = 128
                name = (f'mm-{SHORT[adt]}{SHORT[cdt]}-acc{str(acc).replace("fp", "f")}-{prec}-{M}x{N}x{K}-'
                        f'{bm}x{bn}x{bk}w{nw}s{ns}-t{int(ta)}{int(tb)}-{epi}')
                quick = (ki in (0, 5, 1, 4, 7) and ti in (2, 4, 9) and si in (0, 2)) or name in MM_SEEN_WRONG
                out.append(mm_case(name, adt, bdt, cdt, acc, prec, M, N, K, bm, bn, bk, nw, ns, ta, tb, epi, kind,
                                   acc_in=bool(ti % 2), quick=quick))
    # float data drawn at random: compared within the error of a sum of K products in float32
    for (adt, cdt, prec) in [(torch.float16, torch.float32, 'ieee'), (torch.float32, torch.float32, 'ieee'),
                             (torch.bfloat16, torch.float32, 'ieee'), (torch.float32, torch.float32, 'tf32')]:
        for (bm, bn, bk, nw) in [(64, 64, 32, 4), (128, 128, 32, 8), (32, 32, 16, 1)]:
            M, N, K = 512, 256, 256
            tol = K * 2 ** -23 * 4
            if prec == 'tf32':
                tol = K * 2 ** -10
            out.append(mm_case(f'mmr-{SHORT[adt]}{SHORT[cdt]}-{prec}-{M}x{N}x{K}-{bm}x{bn}x{bk}w{nw}', adt, adt, cdt,
                               tl.float32, prec, M, N, K, bm, bn, bk, nw, 3, kind='rnd', cmp=('tol', 0, tol * 4),
                               quick=bm == 64))
    return out


@triton.jit
def k_attn(q_ptr, k_ptr, v_ptr, o_ptr, sm_scale, N_CTX, HEAD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
           CAUSAL: tl.constexpr):
    """Flash attention, forward (the shape of Triton's tutorial): two products per step and an
    online softmax."""
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    off = bh.to(tl.int64) * N_CTX * HEAD
    rm = start_m * BM + tl.arange(0, BM)
    rd = tl.arange(0, HEAD)
    q = tl.load(q_ptr + off + rm[:, None] * HEAD + rd[None, :], mask=rm[:, None] < N_CTX, other=0)
    m_i = tl.full((BM,), float("-inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, HEAD), tl.float32)
    hi = N_CTX
    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BM, N_CTX)
    for start_n in range(0, hi, BN):
        rn = start_n + tl.arange(0, BN)
        k = tl.load(k_ptr + off + rn[None, :] * HEAD + rd[:, None], mask=rn[None, :] < N_CTX, other=0)
        qk = tl.dot(q, k) * sm_scale
        valid = rn[None, :] < N_CTX
        if CAUSAL:
            valid = valid & (rm[:, None] >= rn[None, :])
        qk = tl.where(valid, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2((qk - m_new[:, None]) * 1.4426950408889634)
        alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_ptr + off + rn[:, None] * HEAD + rd[None, :], mask=rn[:, None] < N_CTX, other=0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(o_ptr + off + rm[:, None] * HEAD + rd[None, :], acc.to(o_ptr.dtype.element_ty), mask=rm[:, None] < N_CTX)


def attn_ref(B, H, N, D, causal):
    def ref(t):
        q, k, v = (t[x].double().reshape(B * H, N, D) for x in 'qkv')
        s = q @ k.transpose(1, 2) / math.sqrt(D)
        if causal:
            s = s.masked_fill(torch.ones(N, N, dtype=torch.bool).triu(1), float('-inf'))
        o = torch.softmax(s, -1) @ v
        res = t['o'].clone()
        res[:B * H * N * D] = o.reshape(-1).to(res.dtype)
        return {'o': res}
    return ref


def attn_cases():
    out = []
    for dt in [torch.float16, torch.bfloat16]:
        for (B, H, N, D, bm, bn, nw, ns, causal) in [(2, 3, 256, 64, 64, 64, 4, 2, False), (1, 4, 1000, 64, 128, 64, 8, 3, True),
                                                     (2, 2, 128, 32, 32, 32, 2, 1, False), (1, 8, 512, 128, 64, 32, 4, 2, True),
                                                     (3, 1, 77, 16, 16, 16, 1, 1, True)]:
            def make(g, B=B, H=H, N=N, D=D, dt=dt):
                t = {x: gen(dt, (B * H * N * D,), 'rnd', g) for x in 'qkv'}
                t['o'] = sentinel(dt, B * H * N * D + 13)
                return t

            def launch(t, interpreted, B=B, H=H, N=N, D=D, bm=bm, bn=bn, nw=nw, ns=ns, causal=causal):
                meta = dict(HEAD=D, BM=bm, BN=bn, CAUSAL=causal)
                if not interpreted:
                    meta.update(num_warps=nw, num_stages=ns)
                k_attn[(triton.cdiv(N, bm), B * H)](t['q'], t['k'], t['v'], t['o'], 1.0 / math.sqrt(D), N, **meta)
            tol = 2 ** -9 if dt == torch.float16 else 2 ** -6
            out.append(Case(f'attn-{SHORT[dt]}-{B}x{H}x{N}x{D}-{bm}x{bn}w{nw}s{ns}c{int(causal)}', k_attn, None, None, make,
                            ['o'], cmp=('tol', tol, tol), ref=attn_ref(B, H, N, D, causal), launch=launch,
                            quick=(N == 256 or N == 1000)))
    return out

# ------------------------------------------------------------------------------------------------
# Memory: masks, gathers, scatters, block pointers, cache hints, 64-bit offsets


@triton.jit
def k_mem(x_ptr, i_ptr, o_ptr, n, m_rows, OP: tl.constexpr, BLOCK: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if OP == "gather":
        idx = tl.load(i_ptr + offs, mask=m, other=0)
        tl.store(o_ptr + offs, tl.load(x_ptr + idx, mask=m), mask=m)
    elif OP == "gather64":
        idx = tl.load(i_ptr + offs, mask=m, other=0).to(tl.int64)
        tl.store(o_ptr + offs, tl.load(x_ptr + idx * 3, mask=m), mask=m)
    elif OP == "scatter":
        idx = tl.load(i_ptr + offs, mask=m, other=0)
        tl.store(o_ptr + idx, tl.load(x_ptr + offs, mask=m), mask=m)
    elif OP == "masked_other":
        mm = m & ((offs % 3) != 1)
        v = tl.load(x_ptr + offs, mask=mm, other=7)
        tl.store(o_ptr + offs, v, mask=m & ((offs % 5) != 2))
    elif OP == "rows":
        # an embedding lookup: whole rows of W columns, at indices loaded from memory
        r = pid * BLOCK + tl.arange(0, BLOCK)
        c = tl.arange(0, W)
        idx = tl.load(i_ptr + r, mask=r < n, other=0)
        v = tl.load(x_ptr + idx[:, None] * W + c[None, :], mask=(r[:, None] < n))
        tl.store(o_ptr + r[:, None] * W + c[None, :], v, mask=r[:, None] < n)
    elif OP == "strided":
        v = tl.load(x_ptr + offs * 3, mask=m)
        tl.store(o_ptr + offs * 2 + 1, v, mask=m)
    elif OP == "reverse":
        v = tl.load(x_ptr + (n - 1 - offs), mask=m)
        tl.store(o_ptr + offs, v, mask=m)
    elif OP == "cache":
        # (ptxas refuses an eviction policy with .ca)
        v = tl.load(x_ptr + offs, mask=m, cache_modifier=".cg")
        w = tl.load(x_ptr + (n - 1 - offs), mask=m, cache_modifier=".ca")
        u = tl.load(x_ptr + offs, mask=m, volatile=True) + tl.load(x_ptr + offs, mask=m, eviction_policy="evict_last")
        tl.store(o_ptr + offs, v, mask=m, cache_modifier=".cs")
        tl.store(o_ptr + n + offs, w, mask=m, cache_modifier=".wt")
        tl.store(o_ptr + 2 * n + offs, u, mask=m, eviction_policy="evict_first")
    elif OP == "far":
        # 64-bit addresses: elements beyond 2**31 of a large buffer
        base = tl.full((BLOCK,), 2147483000, tl.int64) + offs.to(tl.int64) * 2
        v = tl.load(x_ptr + base, mask=m)
        tl.store(o_ptr + offs, v, mask=m)
    elif OP == "checker2d":
        r = pid * BLOCK + tl.arange(0, BLOCK)
        c = tl.arange(0, W)
        mm = (r[:, None] < n) & (((r[:, None] + c[None, :]) % 2) == 0)
        v = tl.load(x_ptr + r[:, None] * W + c[None, :], mask=mm, other=-1)
        tl.store(o_ptr + r[:, None] * W + c[None, :], v, mask=r[:, None] < n)
    else:
        tl.static_assert(False, "op")


def mem_cases():
    out = []
    for dt in [torch.float32, torch.float16, torch.int8, torch.int64, torch.float64, torch.uint16, torch.bfloat16]:
        for op in ['gather', 'gather64', 'scatter', 'masked_other', 'rows', 'strided', 'reverse', 'cache', 'checker2d']:
            for (block, nw, off) in [(1024, 4, 0), (256, 2, 1), (64, 1, 3), (2048, 8, 0)]:
                if (block + len(op) + bits(dt)) % 2 and dt != torch.float32:
                    continue
                n = 100003 if op not in ('rows', 'checker2d') else 3001
                W = 64 if dt != torch.float64 else 32
                rows_block = max(1, block // W)
                blk = rows_block if op in ('rows', 'checker2d') else block
                xsz = {'gather64': 3 * n + 3, 'strided': 3 * n + 3, 'rows': 1000 * W + 1, 'checker2d': n * W}.get(op, n) + off
                osz = {'strided': 2 * n + 9, 'cache': 3 * n + 9, 'rows': n * W + 9, 'checker2d': n * W + 9}.get(op, n + 9)

                def make(g, dt=dt, op=op, n=n, xsz=xsz, osz=osz, off=off):
                    t = {'x': gen(dt, xsz, 'full' if not dt.is_floating_point else 'rnd', g), 'o': sentinel(dt, osz)}
                    if op in ('gather', 'gather64'):
                        t['i'] = torch.randint(0, n, (n + off,), generator=g, dtype=torch.int32)
                    elif op == 'scatter':
                        t['i'] = torch.cat([torch.randperm(n, generator=g).int(), torch.zeros(off, dtype=torch.int32)])
                    elif op == 'rows':
                        t['i'] = torch.randint(0, 1000, (n + off,), generator=g, dtype=torch.int32)
                    else:
                        t['i'] = torch.zeros(1, dtype=torch.int32)
                    return t
                ref = 'interp32' if dt == torch.bfloat16 else 'interp'
                xa = ('view', 'x', off) if off else 'x'
                quick = (block == 1024 and dt in (torch.float16, torch.int64)) or (op == 'rows' and dt == torch.float32 and block == 256)
                out.append(Case(f'mem-{op}-{SHORT[dt]}-b{blk}w{nw}o{off}', k_mem, (triton.cdiv(n, blk),),
                                [xa, 'i', 'o', n, 0], make, ['o'], dict(OP=op, BLOCK=blk, W=W, num_warps=nw), ref=ref,
                                quick=quick))
    # 64-bit offsets into a buffer of more than 2**31 elements (int8: 2 GB on the GPU)
    n = 4096

    def make_far(g):
        x = torch.zeros(2147483000 + 2 * n + 64, dtype=torch.int8)
        x[2147483000:2147483000 + 2 * n] = gen(torch.int8, 2 * n, 'full', g)
        return {'x': x, 'i': torch.zeros(1, dtype=torch.int32), 'o': sentinel(torch.int8, n + 9)}

    def far_ref(t):
        o = t['o'].clone()
        o[:n] = t['x'][2147483000:2147483000 + 2 * n:2]
        return {'o': o}
    out.append(Case('mem-far-i8', k_mem, (n // 256,), ['x', 'i', 'o', n, 0], make_far, ['o'],
                    dict(OP='far', BLOCK=256, W=1, num_warps=4), ref=far_ref, quick=False))
    return out


@triton.jit
def k_blockptr(x_ptr, o_ptr, M, N, sxm, sxn, BM: tl.constexpr, BN: tl.constexpr, OP: tl.constexpr, PAD: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    src = tl.make_block_ptr(x_ptr, (M, N), (sxm, sxn), (pid_m * BM, 0), (BM, BN), (1, 0))
    if OP == "copy":
        dst = tl.make_block_ptr(o_ptr, (M, N), (N, 1), (pid_m * BM, 0), (BM, BN), (1, 0))
        for _ in range(0, tl.cdiv(N, BN)):
            if PAD == "zero":
                v = tl.load(src, boundary_check=(0, 1), padding_option="zero")
            else:
                v = tl.load(src, boundary_check=(0, 1))
            tl.store(dst, v, boundary_check=(0, 1))
            src = tl.advance(src, (0, BN))
            dst = tl.advance(dst, (0, BN))
    else:
        # transpose: rows of x become columns of o
        acc = tl.zeros((BN,), tl.float32)
        for _ in range(0, tl.cdiv(N, BN)):
            v = tl.load(src, boundary_check=(0, 1), padding_option="zero")
            acc += tl.sum(v.to(tl.float32), axis=0)
            dst = tl.make_block_ptr(o_ptr, (N, M), (M, 1), (_ * BN, pid_m * BM), (BN, BM), (1, 0))
            tl.store(dst, tl.trans(v), boundary_check=(0, 1))
            src = tl.advance(src, (0, BN))


def blockptr_cases():
    out = []
    for dt in [torch.float32, torch.float16, torch.int8, torch.float64]:
        for op in ['copy', 'trans']:
            for (bm, bn, nw) in [(32, 32, 4), (64, 16, 2), (16, 128, 4), (128, 64, 8)]:
                M, N = 333, 517

                def make(g, dt=dt, M=M, N=N):
                    return {'x': gen(dt, (M, N), 'full' if not dt.is_floating_point else 'int', g),
                            'o': sentinel(dt, M * N + 17)}
                out.append(Case(f'bptr-{op}-{SHORT[dt]}-{bm}x{bn}w{nw}', k_blockptr, (triton.cdiv(M, bm), 1),
                                ['x', 'o', M, N, N, 1], make, ['o'], dict(BM=bm, BN=bn, OP=op, PAD='zero', num_warps=nw),
                                quick=(bm == 32 and dt != torch.int8)))
    return out

# ------------------------------------------------------------------------------------------------
# Atomics


@triton.jit
def k_atomic(x_ptr, o_ptr, r_ptr, n, nbins, OP: tl.constexpr, SEM: tl.constexpr, SCOPE: tl.constexpr,
             BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    b = offs % nbins
    if OP == "add":
        tl.atomic_add(o_ptr + b, x, mask=m, sem=SEM, scope=SCOPE)
    elif OP == "max":
        tl.atomic_max(o_ptr + b, x, mask=m, sem=SEM, scope=SCOPE)
    elif OP == "min":
        tl.atomic_min(o_ptr + b, x, mask=m, sem=SEM, scope=SCOPE)
    elif OP == "and":
        tl.atomic_and(o_ptr + b, x, mask=m, sem=SEM, scope=SCOPE)
    elif OP == "or":
        tl.atomic_or(o_ptr + b, x, mask=m, sem=SEM, scope=SCOPE)
    elif OP == "xor":
        tl.atomic_xor(o_ptr + b, x, mask=m, sem=SEM, scope=SCOPE)
    elif OP == "slot":
        # each element takes a slot of its own by a counter, and writes itself there: the slots that
        # the counter gives depend on the order, the set of what is written does not
        one = tl.full((BLOCK,), 1, tl.int32)
        s = tl.atomic_add(r_ptr + tl.zeros((BLOCK,), tl.int32), one, mask=m, sem=SEM, scope=SCOPE)
        tl.store(o_ptr + s, x, mask=m)
    elif OP == "xchg":
        old = tl.atomic_xchg(o_ptr + b, x, mask=m, sem=SEM, scope=SCOPE)
        tl.store(r_ptr + offs, old, mask=m)
    elif OP == "addret":
        old = tl.atomic_add(o_ptr + b, x, mask=m, sem=SEM, scope=SCOPE)
        tl.store(r_ptr + offs, old, mask=m)
    else:
        tl.static_assert(False, "op")


@triton.jit
def k_lock(lock_ptr, cnt_ptr, x_ptr, n_iter, SCOPE: tl.constexpr):
    """A critical section by compare-and-swap: one lock, every program; the count is plain loads and
    stores inside it."""
    pid = tl.program_id(0)
    for i in range(0, n_iter):
        while tl.atomic_cas(lock_ptr, 0, 1, sem="acquire", scope=SCOPE) == 1:
            pass
        c = tl.load(cnt_ptr, volatile=True)
        v = tl.load(x_ptr + pid)
        tl.store(cnt_ptr, c + v)
        tl.atomic_xchg(lock_ptr, 0, sem="release", scope=SCOPE)


def sort_key(v):
    if v.dtype.is_floating_point:
        return v.double()
    return v.view(torch.int64) if v.dtype == torch.uint64 else v.to(torch.int64)


def sorted_outs(names):
    """Sort the outputs named (in a key that keeps every bit of the values), the others as they are."""
    def post(o):
        return {k: (v[torch.sort(sort_key(v), stable=True)[1]] if k in names else v) for k, v in o.items()}
    return post


def xchg_post(o):
    import torch
    allv = torch.cat([o['r'], o['o']])
    return {'r': allv[torch.sort(sort_key(allv), stable=True)[1]], 'o': o['o'][:0]}


def atomic_cases():
    out = []
    for dt in [torch.int32, torch.uint32, torch.int64, torch.uint64, torch.float32, torch.float16, torch.float64, torch.bfloat16]:
        for op in ['add', 'max', 'min', 'and', 'or', 'xor', 'slot', 'xchg', 'addret']:
            if op in ('and', 'or', 'xor') and dt.is_floating_point:
                continue
            if op in ('max', 'min', 'xchg') and dt in (torch.float16, torch.bfloat16):
                continue
            for (sem, scope) in [('acq_rel', 'gpu'), ('relaxed', 'gpu'), ('release', 'sys'), ('acquire', 'gpu')]:
                if (len(op) + len(sem) + bits(dt)) % 2 and not (sem == 'acq_rel'):
                    continue
                for (n, nbins, block, nw) in [(1 << 18, 61, 512, 4), (5000, 1, 128, 1), (1 << 16, 4096, 1024, 8)]:
                    if op in ('slot',) and nbins != 61:
                        continue
                    kind = 'full'
                    if op in ('add', 'addret') and dt.is_floating_point:
                        kind = 'small'
                        if dt in (torch.float16, torch.bfloat16) and n / nbins > 100:
                            continue  # a bin's sum would not be exact in 16 bits: its order would show
                    if dt.is_floating_point and op in ('max', 'min', 'xchg', 'slot'):
                        kind = 'rnd'

                    def make(g, dt=dt, op=op, n=n, nbins=nbins, kind=kind):
                        x = gen(dt, n + 3, kind, g)
                        if op == 'addret':
                            # ones: what a bin's adds get back is 0, 1, ... in some order, a set that
                            # the order does not change
                            x = torch.ones(n + 3, dtype=dt)
                        if op == 'slot':
                            o = sentinel(dt, n + 9)
                            r = torch.zeros(4, dtype=torch.int32)
                        else:
                            if op in ('and',):
                                init = torch.full((nbins + 7,), torch.iinfo(dt).max if dt in UINTS else -1, dtype=dt)
                            elif op in ('max',):
                                init = torch.full((nbins + 7,), float('-inf') if dt.is_floating_point else torch.iinfo(dt).min, dtype=dt)
                            elif op in ('min',):
                                init = torch.full((nbins + 7,), float('inf') if dt.is_floating_point else torch.iinfo(dt).max, dtype=dt)
                            else:
                                init = torch.zeros(nbins + 7, dtype=dt)
                            o = init
                            r = sentinel(dt, n + 9)
                        return {'x': x, 'o': o, 'r': r}
                    post, ref = None, ('interp32' if dt == torch.bfloat16 else 'interp')
                    if op == 'slot':
                        post = sorted_outs(['o'])
                    if op == 'addret':
                        post = sorted_outs(['r'])
                    if op == 'xchg':
                        # what each element gets back, with what is left, is every value and the first
                        # ones, in an order that the order of the exchanges decides
                        post = xchg_post
                        if dt.is_floating_point:
                            ref = None  # the interpreter has no exchange of floats
                    out.append(Case(f'atom-{op}-{SHORT[dt]}-{sem}-{scope}-n{n}b{nbins}w{nw}', k_atomic,
                                    (triton.cdiv(n, block),), ['x', 'o', 'r', n, nbins], make,
                                    ['o', 'r'], dict(OP=op, SEM=sem, SCOPE=scope, BLOCK=block, num_warps=nw),
                                    post=post, ref=ref,
                                    quick=(sem == 'acq_rel' and n == 1 << 18 and dt in (torch.float32, torch.int64, torch.float16))))
    for scope in ['gpu', 'sys']:
        for (progs, iters) in [(300, 20), (80, 3)]:
            def make(g, progs=progs):
                return {'lock': torch.zeros(1, dtype=torch.int32), 'cnt': torch.zeros(1, dtype=torch.int32),
                        'x': torch.randint(0, 100, (progs,), generator=g, dtype=torch.int32)}
            out.append(Case(f'atom-lock-{scope}-{progs}x{iters}', k_lock, (progs,), ['lock', 'cnt', 'x', iters], make,
                            ['cnt', 'lock'], dict(SCOPE=scope, num_warps=1), quick=progs == 80))
    return out

# ------------------------------------------------------------------------------------------------
# Control flow, grids, random numbers, odd shapes


@triton.jit
def k_flow(x_ptr, c_ptr, o_ptr, n, OP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if OP == "early":
        if pid % 3 == 1:
            return
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=offs < n) * 2, mask=offs < n)
    elif OP == "while":
        # a trip count read from memory, per program
        cnt = tl.load(c_ptr + pid)
        acc = tl.zeros((BLOCK,), tl.float32)
        i = 0
        while i < cnt:
            offs = (pid * 17 + i) % 64 * BLOCK + tl.arange(0, BLOCK)
            acc += tl.load(x_ptr + offs)
            i += 1
        tl.store(o_ptr + pid * BLOCK + tl.arange(0, BLOCK), acc)
    elif OP == "collatz":
        # a loop that ends when the last lane is done: the number of trips depends on the data
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(c_ptr + offs, mask=offs < n, other=1)
        steps = tl.zeros((BLOCK,), tl.int32)
        while tl.max(v, axis=0) > 1:
            odd = (v % 2) == 1
            nv = tl.where(odd, 3 * v + 1, v // 2)
            act = v > 1
            v = tl.where(act, nv, v)
            steps += act.to(tl.int32)
        tl.store(o_ptr + offs, steps.to(tl.float32), mask=offs < n)
    elif OP == "ifs":
        flag = tl.load(c_ptr + pid)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs, mask=offs < n, other=0)
        if flag > 2:
            if flag > 4:
                x = x * 3
            else:
                x = x - 1
        elif flag == 0:
            x = -x
        else:
            for j in range(0, flag):
                x = x + j
        tl.store(o_ptr + offs, x, mask=offs < n)
    elif OP == "unrolled":
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs, mask=offs < n, other=0)
        acc = x
        for j in tl.static_range(7):
            acc = acc * 0.5 + x * j
        for j in tl.range(0, 9, 2, loop_unroll_factor=2):
            acc = acc + j
        tl.store(o_ptr + offs, acc, mask=offs < n)
    elif OP == "search":
        # a binary search per lane in a sorted table
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        key = tl.load(x_ptr + offs, mask=offs < n, other=0)
        lo = tl.zeros((BLOCK,), tl.int32)
        hi = tl.full((BLOCK,), 4096, tl.int32)
        for _ in range(0, 13):
            mid = (lo + hi) // 2
            v = tl.load(c_ptr + mid, mask=mid < 4096, other=2147483647).to(tl.float32)
            go = v <= key
            lo = tl.where(go, mid + 1, lo)
            hi = tl.where(go, hi, mid)
        tl.store(o_ptr + offs, lo.to(tl.float32), mask=offs < n)
    else:
        tl.static_assert(False, "op")


def flow_cases():
    out = []
    for op in ['early', 'while', 'collatz', 'ifs', 'unrolled', 'search']:
        for (block, nw) in [(128, 1), (512, 4), (1024, 8)]:
            n = 1 << 16
            grid = (triton.cdiv(n, block),) if op not in ('while',) else (300,)

            def make(g, op=op, n=n, block=block):
                if op == 'while':
                    return {'x': gen(torch.float32, 64 * block, 'int', g), 'c': torch.randint(0, 40, (300,), generator=g, dtype=torch.int32),
                            'o': sentinel(torch.float32, 300 * block + 5)}
                if op == 'collatz':
                    return {'x': torch.zeros(1), 'c': torch.randint(1, 3000, (n,), generator=g, dtype=torch.int32),
                            'o': sentinel(torch.float32, n + 5)}
                if op == 'ifs':
                    return {'x': gen(torch.float32, n, 'int', g), 'c': torch.randint(0, 7, (n // block + 1,), generator=g, dtype=torch.int32),
                            'o': sentinel(torch.float32, n + 5)}
                if op == 'search':
                    return {'x': gen(torch.float32, n, 'rnd', g) * 1000,
                            'c': torch.sort(torch.randint(-3000, 3000, (4096,), generator=g, dtype=torch.int32))[0],
                            'o': sentinel(torch.float32, n + 5)}
                return {'x': gen(torch.float32, n, 'int', g), 'c': torch.zeros(1, dtype=torch.int32), 'o': sentinel(torch.float32, n + 5)}
            out.append(Case(f'flow-{op}-b{block}w{nw}', k_flow, grid, ['x', 'c', 'o', n], make, ['o'],
                            dict(OP=op, BLOCK=block, num_warps=nw), quick=block == 512))
    return out


@triton.jit
def k_grid(o_ptr, X: tl.constexpr):
    a = tl.program_id(0)
    b = tl.program_id(1)
    c = tl.program_id(2)
    na = tl.num_programs(0)
    nb = tl.num_programs(1)
    nc = tl.num_programs(2)
    lin = (c.to(tl.int64) * nb + b) * na + a
    v = tl.arange(0, X) + (a * 7 + b * 131 + c * 1031 + na * 3 + nb * 5 + nc * 11)
    tl.store(o_ptr + lin * X + tl.arange(0, X), v)


def grid_cases():
    out = []
    for (ga, gb, gc, x, nw) in [(1, 1, 1, 32, 1), (70000, 1, 1, 16, 1), (3, 65535, 1, 16, 1), (17, 19, 23, 64, 2),
                                (1, 1, 1, 4096, 32), (2, 2, 2, 1024, 16), (200000, 1, 1, 32, 1)]:
        total = ga * gb * gc * x

        def make(g, total=total):
            return {'o': sentinel(torch.int32, total + 13)}

        def ref(t, ga=ga, gb=gb, gc=gc, x=x):
            o = t['o'].clone()
            a = torch.arange(ga).view(1, 1, ga, 1)
            b = torch.arange(gb).view(1, gb, 1, 1)
            c = torch.arange(gc).view(gc, 1, 1, 1)
            v = torch.arange(x).view(1, 1, 1, x) + a * 7 + b * 131 + c * 1031 + ga * 3 + gb * 5 + gc * 11
            o[:ga * gb * gc * x] = v.reshape(-1).int()
            return {'o': o}
        out.append(Case(f'grid-{ga}x{gb}x{gc}-x{x}w{nw}', k_grid, (ga, gb, gc), ['o'], make, ['o'],
                        dict(X=x, num_warps=nw), ref=ref, quick=(ga * gb * gc < 100000)))
    return out


@triton.jit
def k_rand(o_ptr, i_ptr, seed, n, OP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if OP == "randint":
        tl.store(i_ptr + offs, tl.randint(seed, offs), mask=m)
    elif OP == "rand":
        tl.store(o_ptr + offs, tl.rand(seed, offs), mask=m)
    elif OP == "randn":
        tl.store(o_ptr + offs, tl.randn(seed, offs), mask=m)
    elif OP == "randint4x":
        a, b, c, d = tl.randint4x(seed, offs)
        tl.store(i_ptr + offs, a ^ b ^ c ^ d, mask=m)
    elif OP == "dropout":
        x = tl.load(o_ptr + offs, mask=m)
        keep = tl.rand(seed, offs) > 0.3
        tl.store(o_ptr + offs, tl.where(keep, x / 0.7, 0.0), mask=m)
    else:
        tl.static_assert(False, "op")


def rand_cases():
    out = []
    for op in ['randint', 'rand', 'randn', 'randint4x', 'dropout']:
        for seed in [0, 123456789, 2 ** 40 + 5]:
            n = 1 << 17

            def make(g, n=n):
                return {'o': gen(torch.float32, n + 9, 'rnd', g), 'i': sentinel(torch.int32, n + 9)}
            cmp = ('tol', 2 ** -18, 2 ** -18) if op == 'randn' else 'exact'
            if op == 'dropout':
                cmp = ('tol', 2 ** -22, 0)
            out.append(Case(f'rand-{op}-s{seed}', k_rand, (n // 512,), ['o', 'i', seed, n], make, ['o', 'i'],
                            dict(OP=op, BLOCK=512, num_warps=4), cmp=cmp, quick=seed == 123456789))
    return out


@triton.jit
def k_asm(x_ptr, o_ptr, n, OP: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    if OP == "brev":
        r = tl.inline_asm_elementwise("brev.b32 $0, $1;", "=r,r", [x], dtype=tl.uint32, is_pure=True, pack=1)
    elif OP == "prmt":
        r = tl.inline_asm_elementwise("prmt.b32 $0, $1, 0, 0x0123;", "=r,r", [x], dtype=tl.uint32, is_pure=True, pack=1)
    elif OP == "bfind":
        r = tl.inline_asm_elementwise("bfind.u32 $0, $1;", "=r,r", [x], dtype=tl.uint32, is_pure=True, pack=1)
    elif OP == "popc":
        r = tl.inline_asm_elementwise("popc.b32 $0, $1;", "=r,r", [x], dtype=tl.uint32, is_pure=True, pack=1)
    elif OP == "lop3":
        r = tl.inline_asm_elementwise("lop3.b32 $0, $1, 0x0f0f0f0f, 0x33333333, 0xe8;", "=r,r", [x], dtype=tl.uint32,
                                      is_pure=True, pack=1)
    elif OP == "shfl":
        r = tl.inline_asm_elementwise("shfl.sync.bfly.b32 $0, $1, 1, 0x1f, 0xffffffff;", "=r,r", [x], dtype=tl.uint32,
                                      is_pure=False, pack=1)
    else:
        tl.static_assert(False, "op")
    tl.store(o_ptr + offs, r, mask=m)


def asm_ref(op):
    def ref(t):
        x = t['x'][:-1].to(torch.int64) & 0xFFFFFFFF
        vals = x.tolist()
        if op == 'brev':
            r = [int('{:032b}'.format(v)[::-1], 2) for v in vals]
        elif op == 'prmt':
            r = [int.from_bytes(v.to_bytes(4, 'little')[::-1], 'little') for v in vals]
        elif op == 'bfind':
            r = [v.bit_length() - 1 if v else 0xFFFFFFFF for v in vals]
        elif op == 'popc':
            r = [bin(v).count('1') for v in vals]
        elif op == 'lop3':
            r = [((v & 0x0f0f0f0f) | (v & 0x33333333) | (0x0f0f0f0f & 0x33333333)) & 0xFFFFFFFF for v in vals]
            # 0xe8 is the majority of the three
            r = [((v & 0x0f0f0f0f) | (v & 0x33333333) | (0x0f0f0f0f & 0x33333333)) for v in vals]
        o = t['o'].clone()
        o[:len(r)] = torch.tensor(r, dtype=torch.int64).to(torch.uint32)
        return {'o': o}
    return ref


def asm_cases():
    out = []
    for op in ['brev', 'prmt', 'bfind', 'popc', 'lop3', 'shfl']:
        n = 1 << 14

        def make(g, n=n):
            return {'x': gen(torch.uint32, n + 1, 'full', g), 'o': sentinel(torch.uint32, n + 9)}
        out.append(Case(f'asm-{op}', k_asm, (n // 256,), ['x', 'o', n], make, ['o'], dict(OP=op, BLOCK=256, num_warps=2),
                        ref=asm_ref(op) if op != 'shfl' else None, quick=op in ('brev', 'shfl')))
    return out



# ------------------------------------------------------------------------------------------------
# libdevice: the functions of tl.extra.cuda.libdevice (their code is libdevice's, compiled by
# Triton's ptxas: instruction sequences of their own)

from triton.language.extra.cuda import libdevice  # noqa: E402

LD_U = ['acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'cbrt', 'cosh', 'cospi', 'cyl_bessel_i0', 'cyl_bessel_i1', 'erfc', 'erfcinv', 'erfcx', 'erfinv', 'exp10', 'expm1', 'ilogb', 'isinf', 'isnan', 'j0', 'j1', 'lgamma', 'llrint', 'llround', 'log10', 'log1p', 'logb', 'nearbyint', 'normcdf', 'normcdfinv', 'rcbrt', 'rcp_rd', 'rcp_rn', 'rcp_ru', 'rcp_rz', 'rint', 'round', 'signbit', 'sinh', 'sinpi', 'sqrt_rd', 'sqrt_ru', 'sqrt_rz', 'tan', 'tanh', 'tgamma', 'trunc', 'y0', 'y1']
LD_U32 = ['finitef', 'fast_expf', 'fast_logf', 'fast_sinf', 'fast_cosf', 'fast_tanf', 'fast_exp10f', 'fast_log10f', 'fast_log2f', 'saturatef', 'rsqrt_rn', 'float2int_rn', 'float2int_rz', 'float2int_rd', 'float2int_ru', 'float2ll_rn', 'float2ll_rz', 'float2ll_rd', 'float2ll_ru', 'float2uint_rn', 'float2ull_rz', 'float_as_int']
LD_U64 = ['double2float_rn', 'double2float_rz', 'double2float_rd', 'double2float_ru', 'double2int_rn', 'double2int_rz', 'double2int_rd', 'double2int_ru', 'double2ll_rn', 'double2ll_rz', 'double2ll_rd', 'double2ll_ru', 'double2uint_rz', 'double2ull_rn', 'double2loint', 'double2hiint', 'double_as_longlong', 'isfinited', 'rcp64h']
LD_B = ['atan2', 'hypot', 'rhypot', 'fmod', 'remainder', 'copysign', 'nextafter', 'fdim', 'pow', 'add_rz', 'add_rd', 'add_ru', 'mul_rz', 'mul_rd', 'mul_ru', 'div_rz', 'div_rd', 'div_ru', 'sub_rz', 'sub_rd', 'sub_ru']
LD_B32 = ['fast_dividef', 'fast_powf']
LD_T = ['fma_rz', 'fma_rd', 'fma_ru']
LD_I32 = ['clz', 'popc', 'brev', 'ffs', 'int2float_rz', 'int2float_rd', 'int2float_ru', 'int2double_rn', 'uint2float_rz', 'uint2double_rn', 'int_as_float']
LD_I32B = ['mulhi', 'mul24', 'hadd', 'rhadd']
LD_I32T = ['sad', 'byte_perm']
LD_I64 = ['clz', 'popc', 'brev', 'ffs', 'll2float_rz', 'll2float_rd', 'll2float_ru', 'll2double_rz', 'll2double_rd', 'll2double_ru', 'ull2double_rn', 'ull2float_rz', 'longlong_as_double']
LD_I64B = ['mulhi']


@triton.jit
def k_libdevice(x_ptr, y_ptr, z_ptr, o_ptr, n, FN: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    z = tl.load(z_ptr + offs, mask=m)
    if FN == "acos":
        r = libdevice.acos(x)
    elif FN == "acosh":
        r = libdevice.acosh(x)
    elif FN == "asin":
        r = libdevice.asin(x)
    elif FN == "asinh":
        r = libdevice.asinh(x)
    elif FN == "atan":
        r = libdevice.atan(x)
    elif FN == "atanh":
        r = libdevice.atanh(x)
    elif FN == "cbrt":
        r = libdevice.cbrt(x)
    elif FN == "cosh":
        r = libdevice.cosh(x)
    elif FN == "cospi":
        r = libdevice.cospi(x)
    elif FN == "cyl_bessel_i0":
        r = libdevice.cyl_bessel_i0(x)
    elif FN == "cyl_bessel_i1":
        r = libdevice.cyl_bessel_i1(x)
    elif FN == "double2float_rd":
        r = libdevice.double2float_rd(x)
    elif FN == "double2float_rn":
        r = libdevice.double2float_rn(x)
    elif FN == "double2float_ru":
        r = libdevice.double2float_ru(x)
    elif FN == "double2float_rz":
        r = libdevice.double2float_rz(x)
    elif FN == "double2hiint":
        r = libdevice.double2hiint(x)
    elif FN == "double2int_rd":
        r = libdevice.double2int_rd(x)
    elif FN == "double2int_rn":
        r = libdevice.double2int_rn(x)
    elif FN == "double2int_ru":
        r = libdevice.double2int_ru(x)
    elif FN == "double2int_rz":
        r = libdevice.double2int_rz(x)
    elif FN == "double2ll_rd":
        r = libdevice.double2ll_rd(x)
    elif FN == "double2ll_rn":
        r = libdevice.double2ll_rn(x)
    elif FN == "double2ll_ru":
        r = libdevice.double2ll_ru(x)
    elif FN == "double2ll_rz":
        r = libdevice.double2ll_rz(x)
    elif FN == "double2loint":
        r = libdevice.double2loint(x)
    elif FN == "double2uint_rz":
        r = libdevice.double2uint_rz(x)
    elif FN == "double2ull_rn":
        r = libdevice.double2ull_rn(x)
    elif FN == "double_as_longlong":
        r = libdevice.double_as_longlong(x)
    elif FN == "erfc":
        r = libdevice.erfc(x)
    elif FN == "erfcinv":
        r = libdevice.erfcinv(x)
    elif FN == "erfcx":
        r = libdevice.erfcx(x)
    elif FN == "erfinv":
        r = libdevice.erfinv(x)
    elif FN == "exp10":
        r = libdevice.exp10(x)
    elif FN == "expm1":
        r = libdevice.expm1(x)
    elif FN == "fast_cosf":
        r = libdevice.fast_cosf(x)
    elif FN == "fast_exp10f":
        r = libdevice.fast_exp10f(x)
    elif FN == "fast_expf":
        r = libdevice.fast_expf(x)
    elif FN == "fast_log10f":
        r = libdevice.fast_log10f(x)
    elif FN == "fast_log2f":
        r = libdevice.fast_log2f(x)
    elif FN == "fast_logf":
        r = libdevice.fast_logf(x)
    elif FN == "fast_sinf":
        r = libdevice.fast_sinf(x)
    elif FN == "fast_tanf":
        r = libdevice.fast_tanf(x)
    elif FN == "finitef":
        r = libdevice.finitef(x)
    elif FN == "float2int_rd":
        r = libdevice.float2int_rd(x)
    elif FN == "float2int_rn":
        r = libdevice.float2int_rn(x)
    elif FN == "float2int_ru":
        r = libdevice.float2int_ru(x)
    elif FN == "float2int_rz":
        r = libdevice.float2int_rz(x)
    elif FN == "float2ll_rd":
        r = libdevice.float2ll_rd(x)
    elif FN == "float2ll_rn":
        r = libdevice.float2ll_rn(x)
    elif FN == "float2ll_ru":
        r = libdevice.float2ll_ru(x)
    elif FN == "float2ll_rz":
        r = libdevice.float2ll_rz(x)
    elif FN == "float2uint_rn":
        r = libdevice.float2uint_rn(x)
    elif FN == "float2ull_rz":
        r = libdevice.float2ull_rz(x)
    elif FN == "float_as_int":
        r = libdevice.float_as_int(x)
    elif FN == "ilogb":
        r = libdevice.ilogb(x)
    elif FN == "isfinited":
        r = libdevice.isfinited(x)
    elif FN == "isinf":
        r = libdevice.isinf(x)
    elif FN == "isnan":
        r = libdevice.isnan(x)
    elif FN == "j0":
        r = libdevice.j0(x)
    elif FN == "j1":
        r = libdevice.j1(x)
    elif FN == "lgamma":
        r = libdevice.lgamma(x)
    elif FN == "llrint":
        r = libdevice.llrint(x)
    elif FN == "llround":
        r = libdevice.llround(x)
    elif FN == "log10":
        r = libdevice.log10(x)
    elif FN == "log1p":
        r = libdevice.log1p(x)
    elif FN == "logb":
        r = libdevice.logb(x)
    elif FN == "nearbyint":
        r = libdevice.nearbyint(x)
    elif FN == "normcdf":
        r = libdevice.normcdf(x)
    elif FN == "normcdfinv":
        r = libdevice.normcdfinv(x)
    elif FN == "rcbrt":
        r = libdevice.rcbrt(x)
    elif FN == "rcp64h":
        r = libdevice.rcp64h(x)
    elif FN == "rcp_rd":
        r = libdevice.rcp_rd(x)
    elif FN == "rcp_rn":
        r = libdevice.rcp_rn(x)
    elif FN == "rcp_ru":
        r = libdevice.rcp_ru(x)
    elif FN == "rcp_rz":
        r = libdevice.rcp_rz(x)
    elif FN == "rint":
        r = libdevice.rint(x)
    elif FN == "round":
        r = libdevice.round(x)
    elif FN == "rsqrt_rn":
        r = libdevice.rsqrt_rn(x)
    elif FN == "saturatef":
        r = libdevice.saturatef(x)
    elif FN == "signbit":
        r = libdevice.signbit(x)
    elif FN == "sinh":
        r = libdevice.sinh(x)
    elif FN == "sinpi":
        r = libdevice.sinpi(x)
    elif FN == "sqrt_rd":
        r = libdevice.sqrt_rd(x)
    elif FN == "sqrt_ru":
        r = libdevice.sqrt_ru(x)
    elif FN == "sqrt_rz":
        r = libdevice.sqrt_rz(x)
    elif FN == "tan":
        r = libdevice.tan(x)
    elif FN == "tanh":
        r = libdevice.tanh(x)
    elif FN == "tgamma":
        r = libdevice.tgamma(x)
    elif FN == "trunc":
        r = libdevice.trunc(x)
    elif FN == "y0":
        r = libdevice.y0(x)
    elif FN == "y1":
        r = libdevice.y1(x)
    elif FN == "add_rd":
        r = libdevice.add_rd(x, y)
    elif FN == "add_ru":
        r = libdevice.add_ru(x, y)
    elif FN == "add_rz":
        r = libdevice.add_rz(x, y)
    elif FN == "atan2":
        r = libdevice.atan2(x, y)
    elif FN == "copysign":
        r = libdevice.copysign(x, y)
    elif FN == "div_rd":
        r = libdevice.div_rd(x, y)
    elif FN == "div_ru":
        r = libdevice.div_ru(x, y)
    elif FN == "div_rz":
        r = libdevice.div_rz(x, y)
    elif FN == "fast_dividef":
        r = libdevice.fast_dividef(x, y)
    elif FN == "fast_powf":
        r = libdevice.fast_powf(x, y)
    elif FN == "fdim":
        r = libdevice.fdim(x, y)
    elif FN == "fmod":
        r = libdevice.fmod(x, y)
    elif FN == "hypot":
        r = libdevice.hypot(x, y)
    elif FN == "mul_rd":
        r = libdevice.mul_rd(x, y)
    elif FN == "mul_ru":
        r = libdevice.mul_ru(x, y)
    elif FN == "mul_rz":
        r = libdevice.mul_rz(x, y)
    elif FN == "nextafter":
        r = libdevice.nextafter(x, y)
    elif FN == "pow":
        r = libdevice.pow(x, y)
    elif FN == "remainder":
        r = libdevice.remainder(x, y)
    elif FN == "rhypot":
        r = libdevice.rhypot(x, y)
    elif FN == "sub_rd":
        r = libdevice.sub_rd(x, y)
    elif FN == "sub_ru":
        r = libdevice.sub_ru(x, y)
    elif FN == "sub_rz":
        r = libdevice.sub_rz(x, y)
    elif FN == "fma_rd":
        r = libdevice.fma_rd(x, y, z)
    elif FN == "fma_ru":
        r = libdevice.fma_ru(x, y, z)
    elif FN == "fma_rz":
        r = libdevice.fma_rz(x, y, z)
    elif FN == "brev":
        r = libdevice.brev(x)
    elif FN == "clz":
        r = libdevice.clz(x)
    elif FN == "ffs":
        r = libdevice.ffs(x)
    elif FN == "int2double_rn":
        r = libdevice.int2double_rn(x)
    elif FN == "int2float_rd":
        r = libdevice.int2float_rd(x)
    elif FN == "int2float_ru":
        r = libdevice.int2float_ru(x)
    elif FN == "int2float_rz":
        r = libdevice.int2float_rz(x)
    elif FN == "int_as_float":
        r = libdevice.int_as_float(x)
    elif FN == "ll2double_rd":
        r = libdevice.ll2double_rd(x)
    elif FN == "ll2double_ru":
        r = libdevice.ll2double_ru(x)
    elif FN == "ll2double_rz":
        r = libdevice.ll2double_rz(x)
    elif FN == "ll2float_rd":
        r = libdevice.ll2float_rd(x)
    elif FN == "ll2float_ru":
        r = libdevice.ll2float_ru(x)
    elif FN == "ll2float_rz":
        r = libdevice.ll2float_rz(x)
    elif FN == "longlong_as_double":
        r = libdevice.longlong_as_double(x)
    elif FN == "popc":
        r = libdevice.popc(x)
    elif FN == "uint2double_rn":
        r = libdevice.uint2double_rn(x)
    elif FN == "uint2float_rz":
        r = libdevice.uint2float_rz(x)
    elif FN == "ull2double_rn":
        r = libdevice.ull2double_rn(x)
    elif FN == "ull2float_rz":
        r = libdevice.ull2float_rz(x)
    elif FN == "hadd":
        r = libdevice.hadd(x, y)
    elif FN == "mul24":
        r = libdevice.mul24(x, y)
    elif FN == "mulhi":
        r = libdevice.mulhi(x, y)
    elif FN == "rhadd":
        r = libdevice.rhadd(x, y)
    elif FN == "byte_perm":
        r = libdevice.byte_perm(x, y, z & 0x7777)
    elif FN == "sad":
        r = libdevice.sad(x, y, z)
    else:
        tl.static_assert(False, "unknown function")
    tl.store(o_ptr + offs, r, mask=m)


LD_INT_RESULT = {'ilogb': torch.int32, 'signbit': torch.int32, 'isinf': torch.int32, 'isnan': torch.int32,
                 'finitef': torch.int32, 'isfinited': torch.int32, 'llrint': torch.int64, 'llround': torch.int64,
                 'float2int_rn': torch.int32, 'float2int_rz': torch.int32, 'float2int_rd': torch.int32,
                 'float2int_ru': torch.int32, 'float2ll_rn': torch.int64, 'float2ll_rz': torch.int64,
                 'float2ll_rd': torch.int64, 'float2ll_ru': torch.int64, 'float2uint_rn': torch.uint32,
                 'float2ull_rz': torch.uint64, 'float_as_int': torch.int32, 'double2int_rn': torch.int32,
                 'double2int_rz': torch.int32, 'double2int_rd': torch.int32, 'double2int_ru': torch.int32,
                 'double2ll_rn': torch.int64, 'double2ll_rz': torch.int64, 'double2ll_rd': torch.int64,
                 'double2ll_ru': torch.int64, 'double2uint_rz': torch.uint32, 'double2ull_rn': torch.uint64,
                 'double2loint': torch.int32, 'double2hiint': torch.int32, 'double_as_longlong': torch.int64}
LD_DOMAIN = {'acosh': lambda x: 1 + x.abs(), 'atanh': lambda x: torch.tanh(x) * 0.999, 'erfinv': lambda x: torch.tanh(x) * 0.999,
             'erfcinv': lambda x: 1 + torch.tanh(x) * 0.999, 'normcdfinv': lambda x: torch.sigmoid(x),
             'y0': lambda x: x.abs() + 0.01, 'y1': lambda x: x.abs() + 0.01, 'lgamma': lambda x: x.abs() + 0.01,
             'tgamma': lambda x: x.abs() + 0.01, 'log10': lambda x: x.abs() + 1e-3, 'log1p': lambda x: x.abs() - 0.5,
             'float2uint_rn': lambda x: x.abs() * 1000, 'float2ull_rz': lambda x: x.abs() * 1e6,
             'double2uint_rz': lambda x: x.abs() * 1000, 'double2ull_rn': lambda x: x.abs() * 1e9,
             'float2int_rn': lambda x: x * 1000, 'float2int_rz': lambda x: x * 1000, 'float2int_rd': lambda x: x * 1000,
             'float2int_ru': lambda x: x * 1000, 'double2int_rn': lambda x: x * 1000, 'double2int_rz': lambda x: x * 1000,
             'double2int_rd': lambda x: x * 1000, 'double2int_ru': lambda x: x * 1000,
             'float2ll_rn': lambda x: x * 1e6, 'float2ll_rz': lambda x: x * 1e6, 'float2ll_rd': lambda x: x * 1e6,
             'float2ll_ru': lambda x: x * 1e6, 'double2ll_rn': lambda x: x * 1e9, 'double2ll_rz': lambda x: x * 1e9,
             'double2ll_rd': lambda x: x * 1e9, 'double2ll_ru': lambda x: x * 1e9, 'llrint': lambda x: x * 1e6,
             'llround': lambda x: x * 1e6}


def libdevice_cases():
    """Each function on data in its domain; no reference (the interpreter has no libdevice): the
    runs and the two routes are compared."""
    out = []
    plan = []
    for f in LD_U:
        plan += [(f, torch.float32, 1), (f, torch.float64, 1)]
    plan += [(f, torch.float32, 1) for f in LD_U32] + [(f, torch.float64, 1) for f in LD_U64]
    for f in LD_B:
        plan += [(f, torch.float32, 2), (f, torch.float64, 2)]
    plan += [(f, torch.float32, 2) for f in LD_B32]
    for f in LD_T:
        plan += [(f, torch.float32, 3), (f, torch.float64, 3)]
    plan += [(f, torch.int32, 1) for f in LD_I32] + [(f, torch.int32, 2) for f in LD_I32B]
    plan += [(f, torch.int32, 3) for f in LD_I32T] + [(f, torch.int64, 1) for f in LD_I64]
    plan += [(f, torch.int64, 2) for f in LD_I64B]
    for f, dt, arity in plan:
        n = (1 << 15) - 7
        if f in LD_INT_RESULT:
            odt = LD_INT_RESULT[f]
        elif f.startswith('double2float'):
            odt = torch.float32
        elif f in ('int2float_rz', 'int2float_rd', 'int2float_ru', 'uint2float_rz', 'll2float_rz', 'll2float_rd',
                   'll2float_ru', 'ull2float_rz', 'int_as_float'):
            odt = torch.float32
        elif f in ('int2double_rn', 'uint2double_rn', 'll2double_rz', 'll2double_rd', 'll2double_ru', 'ull2double_rn',
                   'longlong_as_double'):
            odt = torch.float64
        else:
            odt = dt

        def make(g, f=f, dt=dt, n=n, odt=odt):
            if dt.is_floating_point:
                xs = [gen(torch.float64, n + 1, 'rnd', g) * 4 for _ in range(3)]
                if f in LD_DOMAIN:
                    xs[0] = LD_DOMAIN[f](xs[0])
                t = {k: v.to(dt) for k, v in zip('xyz', xs)}
            else:
                t = {k: gen(dt, n + 1, 'full', g) for k in 'xyz'}
            t['o'] = sentinel(odt, n + 11)
            return t
        quick = f in ('erfinv', 'tgamma', 'fmod', 'pow', 'brev', 'mulhi', 'double2float_ru', 'float2ll_rd', 'fma_rz', 'll2float_rz')
        out.append(Case(f'ld-{f}-{SHORT[dt]}', k_libdevice, (triton.cdiv(n, 256),), ['x', 'y', 'z', 'o', n], make, ['o'],
                        dict(FN=f, BLOCK=256, num_warps=4), ref=None, quick=quick and dt in (torch.float64, torch.int64, torch.int32)))
    return out

# ------------------------------------------------------------------------------------------------

FAMILIES = [binary_cases, ternary_cases, unary_cases, convert_cases, reduce_cases, scan_cases, sort_cases, mm_cases,
            attn_cases, mem_cases, blockptr_cases, atomic_cases, flow_cases, grid_cases, rand_cases, asm_cases,
            libdevice_cases]


def cases(level='full'):
    out = []
    for f in FAMILIES:
        out += f()
    names = [c.name for c in out]
    assert len(names) == len(set(names)), 'two cases of one name'
    if level == 'quick':
        out = [c for c in out if c.quick]
    return out
