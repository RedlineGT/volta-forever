#!/usr/bin/env python3
"""Operations of PyTorch on the GPU against the same operations on the CPU.

    torch-ops.py [-v] [PATTERN...]        (run by tests/gpu/torch.sh, which sets the paths)
    torch-ops.py --comparison             (what decides ok, checked on tensors of the CPU)

A stock PyTorch wheel has machine code only, for sm_75 and later: on a V100 every kernel it launches
went through the translator of the layer. Each case computes on the CPU and on the GPU from the same
inputs and compares: equal for integers and booleans, within a tolerance for floats (the CPU and the
GPU do not round every function alike). A case ends as

    ok        the same result
    DIFFERENT another result: the translation is wrong, or the tolerance is too tight
    refused   a kernel it needs is not given to the program: its function is not translated (the
              reasons are counted at the end, from what the proxy says)
    error     PyTorch raised something else on the GPU
    CRASH     a fault on the GPU, after which the process can do nothing more there, or the process
              ended: the cases go on in a new process
    (skipped: the CPU cannot do it, so there is nothing to compare with; why is counted at the
    end, so that a case skipped for a mistake of its own is seen)

The cases run in a worker process, restarted after the one that ends it. With PATTERN, only the cases
whose name contains one of them; -v prints every case. The exit status is 1 if a case is DIFFERENT
or a CRASH.

A worker writes its verdicts on a descriptor of its own: kernels print on standard output, and a
message that runs into a verdict would hide it. What the workers print on standard output and on
standard error goes to two files beside the proxy log. A case that has not ended after
$VF_CASE_WAIT seconds (default 600) is stopped and counted as a CRASH. The exit status is 1 too if
the run itself went wrong: a worker wrote a line that is not a verdict, the verdicts are not as many
as the cases, or a worker ended, or had to be stopped, outside a case; 2 if no case could be run.
"""
import os, re, select, signal, subprocess, sys, tempfile

VERDICTS = ('ok', 'DIFFERENT', 'refused', 'error', 'CRASH', 'skipped')


def build():
    """The cases: (name, function of the device that returns a tensor or several)."""
    import torch
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(20261116)
    floats = {'f32': torch.float32, 'f64': torch.float64, 'f16': torch.float16}
    ints = {'i32': torch.int32, 'i64': torch.int64, 'u8': torch.uint8, 'i8': torch.int8, 'i16': torch.int16}
    cases = []

    def case(name, fn, *inputs):
        cases.append((name, fn, inputs))

    def rnd(dtype, *shape, low=None):
        if dtype.is_floating_point:
            t = torch.randn(*shape, generator=g, dtype=torch.float64)
            if low is not None:
                t = t.abs() + low
            return t.to(dtype)
        if dtype == torch.bool:
            return torch.rand(*shape, generator=g) < 0.5
        lo, hi = (0, 100) if dtype == torch.uint8 else (-100, 100)
        if low is not None:
            lo = low
        return torch.randint(lo, hi, shape, generator=g).to(dtype)

    # ---- one tensor in, one out, by element ---------------------------------------------------------------
    unary = '''abs neg exp expm1 log log1p log2 log10 sqrt rsqrt sin cos tan asin acos atan sinh cosh tanh
        asinh acosh atanh sigmoid erf erfc erfinv floor ceil round trunc frac sign sgn reciprocal square exp2
        lgamma digamma isnan isinf isfinite logical_not relu positive signbit nan_to_num i0 sinc deg2rad
        rad2deg angle'''.split()
    for tag, dtype in floats.items():
        a = rnd(dtype, 65, 33)
        for op in unary:
            case('%s.%s' % (op, tag), lambda x, op=op: getattr(torch, op)(x), a)
        case('gelu.%s' % tag, F.gelu, a)
        case('silu.%s' % tag, F.silu, a)
        case('elu.%s' % tag, F.elu, a)
        case('leaky_relu.%s' % tag, F.leaky_relu, a)
        case('hardtanh.%s' % tag, F.hardtanh, a)
        case('softplus.%s' % tag, F.softplus, a)
        case('logsigmoid.%s' % tag, F.logsigmoid, a)
        case('hardswish.%s' % tag, F.hardswish, a)
        case('mish.%s' % tag, F.mish, a)
        case('clamp.%s' % tag, lambda x: x.clamp(-0.5, 0.5), a)
        case('pow_scalar.%s' % tag, lambda x: x.abs().pow(1.7), a)
        case('mul_scalar.%s' % tag, lambda x: x * 2 + 1, a)
    for tag, dtype in ints.items():
        a = rnd(dtype, 65, 33)
        for op in 'abs neg sign bitwise_not square logical_not'.split():
            case('%s.%s' % (op, tag), lambda x, op=op: getattr(torch, op)(x), a)
        case('clamp.%s' % tag, lambda x: x.clamp(3, 40), a)
        case('shift.%s' % tag, lambda x: (x << 2, x >> 1), a)

    # ---- two tensors in -------------------------------------------------------------------------------------
    binary = '''add sub mul div maximum minimum fmax fmin fmod remainder atan2 hypot copysign eq ne lt le gt ge
        logical_and logical_or logical_xor xlogy nextafter heaviside floor_divide true_divide'''.split()
    for tag, dtype in floats.items():
        a, b = rnd(dtype, 65, 33), rnd(dtype, 65, 33)
        for op in binary:
            case('%s.%s' % (op, tag), lambda x, y, op=op: getattr(torch, op)(x, y), a, b)
        case('pow.%s' % tag, lambda x, y: x.abs().pow(y), a, b)
        case('broadcast_add.%s' % tag, lambda x, y: x + y[:1] * y[:, :1], a, b)
        case('where.%s' % tag, lambda x, y: torch.where(x > y, x, y * 2), a, b)
        case('lerp.%s' % tag, lambda x, y: torch.lerp(x, y, 0.3), a, b)
        case('addcmul.%s' % tag, lambda x, y: torch.addcmul(x, x, y, value=0.5), a, b)
        case('addcdiv.%s' % tag, lambda x, y: torch.addcdiv(x, x, y.abs() + 1, value=0.5), a, b)
        case('inplace.%s' % tag, lambda x, y: x.clone().mul_(y).add_(1).sub_(y).div_(3), a, b)
    for tag, dtype in ints.items():
        a, b = rnd(dtype, 65, 33), rnd(dtype, 65, 33, low=1)
        for op in '''add sub mul maximum minimum eq ne lt le gt ge bitwise_and bitwise_or bitwise_xor
                floor_divide remainder fmod gcd lcm'''.split():
            case('%s.%s' % (op, tag), lambda x, y, op=op: getattr(torch, op)(x, y), a, b)
        case('div.%s' % tag, lambda x, y: torch.div(x, y), a, b)
        case('where.%s' % tag, lambda x, y: torch.where(x > y, x, y), a, b)
    bools = rnd(torch.bool, 65, 33), rnd(torch.bool, 65, 33)
    for op in 'logical_and logical_or logical_xor bitwise_and bitwise_or bitwise_xor eq ne'.split():
        case('%s.bool' % op, lambda x, y, op=op: getattr(torch, op)(x, y), *bools)

    # ---- conversions ----------------------------------------------------------------------------------------
    every = dict(floats, **ints, bool=torch.bool)
    for tag, dtype in every.items():
        a = rnd(dtype, 65, 33)
        for to, target in every.items():
            if to != tag:
                case('to_%s.%s' % (to, tag), lambda x, target=target: x.to(target), a)

    # ---- reductions and scans -------------------------------------------------------------------------------
    for tag, dtype in dict(floats, i32=torch.int32, i64=torch.int64).items():
        a = rnd(dtype, 37, 129)
        small = rnd(dtype, 6, 7) if dtype.is_floating_point else rnd(dtype, 6, 7).clamp(-3, 3)
        case('sum.%s' % tag, lambda x: (x.sum(), x.sum(0), x.sum(1)), a)
        case('prod.%s' % tag, lambda x: (x.prod(0), x.prod(1)), small)
        case('amax_amin.%s' % tag, lambda x: (x.amax(), x.amin(0), x.amax(1)), a)
        case('max_dim.%s' % tag, lambda x: x.max(1), a)
        case('min_dim.%s' % tag, lambda x: x.min(0), a)
        case('argmax_argmin.%s' % tag, lambda x: (x.argmax(), x.argmax(1), x.argmin(0)), a)
        case('any_all.%s' % tag, lambda x: ((x > 0).any(1), (x > -5).all(0), x.any(), x.all()), a)
        case('cumsum.%s' % tag, lambda x: (x.cumsum(0), x.cumsum(1)), a)
        case('cumprod.%s' % tag, lambda x: x.cumprod(1), small)
        case('cummax_cummin.%s' % tag, lambda x: (x.cummax(1), x.cummin(0)), a)
        case('count_nonzero.%s' % tag, lambda x: (torch.count_nonzero(x > 0), torch.count_nonzero(x, 1)), a)
        case('sort.%s' % tag, lambda x: x.sort(dim=1, stable=True), a)
        case('sort_descending.%s' % tag, lambda x: x.sort(dim=0, descending=True, stable=True), a)
        case('argsort.%s' % tag, lambda x: x.argsort(dim=1, stable=True), a)
        case('topk.%s' % tag, lambda x: x.topk(5, 1).values, a)
        case('kthvalue.%s' % tag, lambda x: x.kthvalue(7, 1).values, a)
        case('unique.%s' % tag, lambda x: torch.unique(x if x.dtype.is_floating_point else x % 17), a)
        case('median.%s' % tag, lambda x: x.median(1).values, a)
        if dtype.is_floating_point:
            case('mean.%s' % tag, lambda x: (x.mean(), x.mean(0), x.mean(1)), a)
            case('std_var.%s' % tag, lambda x: (x.std(1), x.var(0)), a)
            case('norm.%s' % tag, lambda x: (x.norm(), x.norm(1, 1), torch.linalg.vector_norm(x, 3.0, 0)), a)
            case('logsumexp.%s' % tag, lambda x: x.logsumexp(1), a)
            case('softmax.%s' % tag, lambda x: (x.softmax(1), x.log_softmax(0)), a)
            case('nansum.%s' % tag, lambda x: torch.where(x > 1, x * float('nan'), x).nansum(1), a)

    # ---- indices ----------------------------------------------------------------------------------------------
    for tag, dtype in dict(f32=torch.float32, f64=torch.float64, f16=torch.float16, i64=torch.int64, u8=torch.uint8).items():
        a, b = rnd(dtype, 40, 50), rnd(dtype, 40, 50)
        rows = torch.randint(0, 40, (64,), generator=g)
        cols = torch.randint(0, 50, (40, 20), generator=g)
        mask = torch.rand(40, 50, generator=g) < 0.3
        once = torch.randperm(50, generator=g)[:20].expand(40, 20).contiguous()
        case('index_select.%s' % tag, lambda x, i: (x.index_select(0, i), x.index_select(1, i % 50)), a, rows)
        case('gather.%s' % tag, lambda x, i: x.gather(1, i), a, cols)
        case('take_along_dim.%s' % tag, lambda x, i: torch.take_along_dim(x, i, 1), a, cols)
        case('scatter.%s' % tag, lambda x, y, i: x.scatter(1, i, y[:, :20]), a, b, once)
        case('scatter_add.%s' % tag, lambda x, y, i: x.scatter_add(1, i, y[:, :20]), a, b, cols)
        case('index_add.%s' % tag, lambda x, y, i: x.index_add(0, i, y[:1].expand(64, 50).contiguous()), a, b, rows)
        case('index_fill.%s' % tag, lambda x, i: x.index_fill(0, i, 3), a, rows)
        case('masked_fill.%s' % tag, lambda x, m: x.masked_fill(m, 7), a, mask)
        case('masked_select.%s' % tag, lambda x, m: x.masked_select(m), a, mask)
        case('masked_scatter.%s' % tag, lambda x, y, m: x.masked_scatter(m, y), a, b, mask)
        case('nonzero.%s' % tag, lambda x, m: (x * m).nonzero(), a, mask)
        case('getitem_index.%s' % tag, lambda x, i: x[i], a, rows)
        case('getitem_two.%s' % tag, lambda x, i: x[i, (i * 7) % 50], a, rows)
        case('getitem_mask.%s' % tag, lambda x, m: x[m], a, mask)
        case('setitem_mask.%s' % tag, lambda x, m: setitem(x, m, 5), a, mask)
        case('setitem_index.%s' % tag, lambda x, y, i: setitem(x, torch.unique(i), y[:1]), a, b, rows)
        case('take.%s' % tag, lambda x, i: x.take(i * 31), a, rows)
        case('cat_stack.%s' % tag, lambda x, y: (torch.cat([x, y], 0), torch.cat([x, y], 1), torch.stack([x, y], 1)), a, b)
        case('flip_roll.%s' % tag, lambda x: (x.flip(0), x.flip(0, 1), x.roll(3, 1), x.roll((1, -2), (0, 1))), a)
        case('tril_triu.%s' % tag, lambda x: (x.tril(), x.triu(1), x.tril(-2)), a)
        case('repeat.%s' % tag, lambda x: (x.repeat(2, 3), x.repeat_interleave(2, 0), x[:1].expand(5, 50).contiguous()), a)
        case('permute.%s' % tag, lambda x: (x.t().contiguous(), x.reshape(4, 10, 50).permute(2, 0, 1).contiguous()), a)
        case('strided.%s' % tag, lambda x: (x[::2, 1::3].contiguous(), x[5:, :7].clone(), x.unfold(1, 5, 3).contiguous()), a)
        case('pad.%s' % tag, lambda x: F.pad(x, (1, 2, 3, 0), value=1), a)
        case('diag.%s' % tag, lambda x: (x.diagonal().clone(), torch.diag(x[0]), x[:, :40].trace()), a)
        case('fill.%s' % tag, lambda x: (torch.full_like(x, 3), torch.zeros_like(x), x.clone().fill_(2)), a)
    edges = torch.linspace(-2, 2, 33)
    values = rnd(torch.float32, 2000)
    case('searchsorted.f32', lambda e, v: (torch.searchsorted(e, v), torch.bucketize(v, e, right=True)), edges, values)
    case('histc.f32', lambda v: torch.histc(v, 20, -3, 3), values)
    case('bincount.i64', lambda v: torch.bincount(v, minlength=100), torch.randint(0, 100, (5000,), generator=g))
    case('one_hot.i64', lambda v: F.one_hot(v, 12), torch.randint(0, 12, (300,), generator=g))
    case('embedding.f32', lambda w, i: F.embedding(i, w), rnd(torch.float32, 100, 16), torch.randint(0, 100, (7, 9), generator=g))

    # ---- made on the device ---------------------------------------------------------------------------------
    for tag, dtype in dict(f32=torch.float32, f64=torch.float64, i64=torch.int64, i32=torch.int32).items():
        case('arange.%s' % tag, lambda d, dtype=dtype: (torch.arange(1000, dtype=dtype, device=d), torch.arange(5, 400, 7, dtype=dtype, device=d)), None)
        case('eye_full.%s' % tag, lambda d, dtype=dtype: (torch.eye(17, 23, dtype=dtype, device=d), torch.full((9, 9), 4, dtype=dtype, device=d)), None)
    case('linspace.f32', lambda d: (torch.linspace(-1, 1, 257, device=d), torch.logspace(0, 3, 50, device=d)), None)
    case('tri_indices', lambda d: (torch.tril_indices(13, 17, device=d), torch.triu_indices(13, 17, 1, device=d)), None)
    case('randperm.sorted', lambda d: torch.randperm(1000, device=d).sort().values, None)
    case('rand.range', lambda d: stats(torch.rand(200000, device=d)), None)
    case('randn.moments', lambda d: stats(torch.randn(200000, device=d))[:2], None)
    case('randint.range', lambda d: stats(torch.randint(0, 10, (200000,), device=d).float()), None)

    # ---- layers that PyTorch computes itself ----------------------------------------------------------------
    for tag, dtype in dict(f32=torch.float32, f64=torch.float64).items():
        x = rnd(dtype, 4, 6, 20, 24)
        seq = rnd(dtype, 8, 30, 16)
        target = torch.randint(0, 16, (8, 30), generator=g)
        case('max_pool2d.%s' % tag, lambda t: F.max_pool2d(t, 3, 2, 1), x)
        case('avg_pool2d.%s' % tag, lambda t: F.avg_pool2d(t, 2), x)
        case('adaptive_pool.%s' % tag, lambda t: (F.adaptive_avg_pool2d(t, (5, 7)), F.adaptive_max_pool2d(t, 3)), x)
        case('interpolate.%s' % tag, lambda t: (F.interpolate(t, scale_factor=2, mode='nearest'), F.interpolate(t, size=(31, 17), mode='bilinear', align_corners=False)), x)
        case('pixel_shuffle.%s' % tag, lambda t: F.pixel_shuffle(t[:, :4], 2), x)
        case('pad_reflect.%s' % tag, lambda t: (F.pad(t, (2, 3, 1, 2), mode='reflect'), F.pad(t, (2, 3, 1, 2), mode='replicate')), x)
        case('layer_norm.%s' % tag, lambda t: F.layer_norm(t, (16,)), seq)
        case('group_norm.%s' % tag, lambda t: F.group_norm(t, 3), x)
        case('batch_norm.%s' % tag, lambda t: F.batch_norm(t, t.mean((0, 2, 3)), t.var((0, 2, 3)) + 1, training=False), x)
        case('cross_entropy.%s' % tag, lambda t, y: (F.cross_entropy(t.reshape(-1, 16), y.reshape(-1)), F.nll_loss(t.log_softmax(-1).reshape(-1, 16), y.reshape(-1), reduction='none')), seq, target)
        case('losses.%s' % tag, lambda t: (F.mse_loss(t, t.roll(1, 0)), F.l1_loss(t, t.flip(1)), F.smooth_l1_loss(t, t.flip(2)), F.binary_cross_entropy_with_logits(t, (t > 0).to(t.dtype))), seq)
        case('dropout_off.%s' % tag, lambda t: F.dropout(t, 0.5, training=False), seq)
        case('grid_sample.%s' % tag, lambda t: F.grid_sample(t, torch.stack(torch.meshgrid(torch.linspace(-1, 1, 9, dtype=t.dtype, device=t.device), torch.linspace(-1, 1, 11, dtype=t.dtype, device=t.device), indexing='ij'), -1).flip(-1).expand(4, 9, 11, 2), align_corners=False), x)
        case('outer_cross.%s' % tag, lambda t: (torch.outer(t[0, 0, 0], t[0, 0, 1]), torch.linalg.cross(t[..., :3], t[..., 3:6])), x)
        case('complex.%s' % tag, lambda t: torch.view_as_real(torch.view_as_complex(t.reshape(4, 6, 20, 12, 2).contiguous()) * (1 + 2j)), x)

    # ---- at the sizes of real models: PyTorch has other kernels for them than for small tensors ---------------
    # (The first battery had none of these, and said nothing of the softmax of a row of 50,000
    # logits, which its kernel for long rows could not do: tests/gpu/torch-bench.sh found it.)
    for tag, dtype in floats.items():
        rows = rnd(dtype, 96, 3000)
        case('long.softmax.%s' % tag, lambda t: (t.softmax(1), t.log_softmax(1)), rows)
        case('long.topk.%s' % tag, lambda t: (t.topk(40, 1).values, t.topk(700, 1).values, t.kthvalue(11, 1).values), rows)
        case('long.sort.%s' % tag, lambda t: (t.sort(dim=1).values, t.sort(dim=1, descending=True, stable=True).values), rows)
        case('long.reduce.%s' % tag, lambda t: (t.sum(1), t.max(1).values, t.min(0).values, t.argmax(1), t.logsumexp(1), t.amax(1), t.any(1)), rows)
        # Running sums: of -1, 0 and 1, which are exact in any order and in halves too, and of the
        # values themselves where the order of the additions does not show (in halves it does: the
        # sm_70 build of PyTorch differs from the CPU there as much as the translation).
        case('long.cumsum.%s' % tag, lambda t, steps: (steps.cumsum(1),) + (() if t.dtype == torch.float16 else (t.cumsum(1),)), rows, rows.sign())
        case('long.cross_entropy.%s' % tag, lambda t, y: F.cross_entropy(t, y), rows, torch.randint(0, 3000, (96,), generator=g))
    for tag, dtype in dict(i32=torch.int32, i64=torch.int64).items():
        counts = torch.randint(0, 40, (96, 3000), generator=g).to(dtype)
        case('long.mode.%s' % tag, lambda t: (t.mode(1).values, t.median(1).values, t.sum(1), t.amax(1), t.cummax(1).values), counts)
        case('long.unique.%s' % tag, lambda t: (t.unique(), t.bincount(minlength=40) if t.dim() == 1 else t.flatten().bincount(minlength=40)), counts)
    for tag, dtype in dict(f32=torch.float32, f16=torch.float16).items():
        heads = rnd(dtype, 2, 8, 200, 64)
        case('attention.%s' % tag, lambda t: F.scaled_dot_product_attention(t, t.flip(2), t.roll(1, 2)), heads)
        case('attention.causal.%s' % tag, lambda t: F.scaled_dot_product_attention(t, t.flip(2), t.roll(1, 2), is_causal=True), heads)
        # Other sizes of a head (PyTorch has a kernel for each range), a mask that is added, and
        # the gradients (the kernels of the way back are others).
        for width in (16, 32, 128, 256):
            wide = rnd(dtype, 2, 4, 96, width)
            case('attention.%d.%s' % (width, tag), lambda t: F.scaled_dot_product_attention(t, t.flip(2), t.roll(1, 2)), wide)
        mask = torch.where(torch.rand(200, 200, generator=g) < 0.2, float('-inf'), 0.0).to(dtype)
        mask[torch.arange(200), torch.arange(200)] = 0
        case('attention.masked.%s' % tag, lambda t, m: F.scaled_dot_product_attention(t, t.flip(2), t.roll(1, 2), attn_mask=m), heads, mask)
        def gradients(t):
            q, k, v = (x.detach().clone().requires_grad_(True) for x in (t, t.flip(2), t.roll(1, 2)))
            out = F.scaled_dot_product_attention(q, k, v)
            (out.float() * out.float()).sum().backward()
            return q.grad, k.grad, v.grad
        case('attention.grad.%s' % tag, gradients, rnd(dtype, 2, 4, 64, 64))
    weights = [rnd(torch.float32, 300, 40), rnd(torch.float32, 1000), rnd(torch.float32, 7, 5, 3, 3)]
    def step(a, b, c):
        """What an optimizer does to a list of parameters at once (the `foreach` kernels)."""
        params, grads = [a.clone(), b.clone(), c.clone()], [a.flip(0), b.roll(3), c.neg()]
        torch._foreach_mul_(params, 0.9)
        torch._foreach_add_(params, grads, alpha=-0.01)
        torch._foreach_addcmul_(params, grads, grads, value=0.5)
        torch._foreach_addcdiv_(params, grads, [x.abs() + 1 for x in params], value=-0.1)
        return tuple(params) + tuple(torch._foreach_sqrt([x.abs() for x in params]))
    case('optimizer.foreach.f32', step, *weights)
    images = rnd(torch.float32, 3, 5, 20, 24)
    case('unfold.f32', lambda t: (F.unfold(t, 3, padding=1), F.fold(F.unfold(t, 2, stride=2), (20, 24), 2, stride=2)), images)

    # ---- through NVIDIA's libraries (cuBLAS, cuSOLVER, cuFFT, cuDNN): their kernels, not PyTorch's -------------
    for tag, dtype in dict(f32=torch.float32, f64=torch.float64, f16=torch.float16).items():
        a, b, v = rnd(dtype, 48, 56), rnd(dtype, 56, 40), rnd(dtype, 56)
        case('lib.matmul.%s' % tag, lambda x, y: x @ y, a, b)
        case('lib.mv_dot.%s' % tag, lambda x, y: (x @ y, y.dot(y)), a, v)
        case('lib.bmm.%s' % tag, lambda x, y: torch.bmm(x.expand(3, 48, 56).contiguous(), y.expand(3, 56, 40).contiguous()), a, b)
        case('lib.linear.%s' % tag, lambda x, y, z: F.linear(x, y.t(), z[:40]), a, b, v)
    x = rnd(torch.float32, 2, 3, 20, 24)
    case('lib.conv2d.f32', lambda t, w: F.conv2d(t, w, padding=1), x, rnd(torch.float32, 5, 3, 3, 3))
    case('lib.conv1d.f32', lambda t, w: F.conv1d(t[:, :, 0], w, stride=2), x, rnd(torch.float32, 4, 3, 5))
    case('lib.fft.f32', lambda t: torch.view_as_real(torch.fft.rfft(t, dim=-1)), x)
    # In double precision, and for lengths that are powers of 7 or have a large prime factor, cuFFT
    # has kernels that take the address of a table of the library: the driver writes it into their
    # code when it loads them (relocations, which must follow the instructions they are on when the
    # code is translated).
    for n in (64, 343, 97, 1021):
        case('lib.fft.%d.c128' % n, lambda t: torch.view_as_real(torch.fft.fft(torch.view_as_complex(t), dim=-1)), rnd(torch.float64, 3, n, 2))
    case('lib.fft.1715.c64', lambda t: torch.view_as_real(torch.fft.ifft(torch.view_as_complex(t), dim=-1)), rnd(torch.float32, 3, 1715, 2))
    case('lib.solve.f64', lambda m, r: torch.linalg.solve(m @ m.t() + 30 * torch.eye(30, dtype=m.dtype, device=m.device), r), rnd(torch.float64, 30, 30), rnd(torch.float64, 30, 4))
    case('lib.cholesky.f64', lambda m: torch.linalg.cholesky(m @ m.t() + 30 * torch.eye(30, dtype=m.dtype, device=m.device)), rnd(torch.float64, 30, 30))
    # ---- sorts of keys of one byte in long rows, and of long vectors ------------------------------------------
    # (At the end, so that the cases before keep the random inputs they always had.)
    # (Kernels that name every register of the machine, or nearly: PyTorch's radix sort of a row of
    # 513 to 4,096 keys with their places, cub's for a vector of keys of one or two bytes. Refused
    # for want of registers as long as the translation kept its own values in registers from one
    # instruction to the next: `docs/translator.md`, "Registers".)
    for tag, dtype in dict(u8=torch.uint8, i8=torch.int8, bool=torch.bool).items():
        for n in (300, 600, 1500, 3000):
            rows, columns, middle = rnd(dtype, 12, n), rnd(dtype, n, 12), rnd(dtype, 3, n, 4)
            case('rows.sort.%d.%s' % (n, tag), lambda t: (t.sort(dim=1, stable=True), t.sort(dim=1, descending=True, stable=True)), rows)
            case('rows.sort.columns.%d.%s' % (n, tag), lambda t: t.sort(dim=0, stable=True), columns)
            case('rows.sort.middle.%d.%s' % (n, tag), lambda t: t.sort(dim=1, stable=True), middle)
    for tag, dtype in dict(u8=torch.uint8, i8=torch.int8, bool=torch.bool, i16=torch.int16).items():
        vector = rnd(dtype, 100000)
        case('vector.sort.%s' % tag, lambda t: (t.sort(stable=True), t.sort(descending=True, stable=True)), vector)
        case('vector.unique.%s' % tag, lambda t: torch.unique(t, return_inverse=True, return_counts=True), vector)
        case('vector.mode.%s' % tag, lambda t: (t.reshape(20, 5000).mode(1).values, t.reshape(20, 5000).median(1).values), vector)
    # ---- gradients through attention for wide heads ------------------------------------------------------------
    # (PyTorch has a kernel of the way back for each range of head size: to 32, to 64, to 128, above.
    # For float32 the last two name every register of the machine and were refused; so were their
    # forms for tensors that do not start at a multiple of 16 bytes. The cases before had heads of
    # 64 only.)
    def wide_gradients(t, unaligned):
        def leaf(x):
            if not unaligned:
                return x.detach().clone().requires_grad_(True)
            # The same values in a tensor that starts 4 bytes into its storage.
            room = torch.empty(x.numel() + 1, dtype=x.dtype, device=x.device)
            view = room[1:].view(x.shape)
            view.copy_(x)
            return view.requires_grad_(True)
        q, k, v = (leaf(x) for x in (t, t.flip(2), t.roll(1, 2)))
        out = F.scaled_dot_product_attention(q, k, v)
        (out.float() * out.float()).sum().backward()
        return q.grad, k.grad, v.grad
    for tag, dtype in dict(f32=torch.float32, f16=torch.float16).items():
        for width in (96, 128, 192, 256):
            heads = rnd(dtype, 2, 2, 48, width)
            case('attention.grad.%d.%s' % (width, tag), lambda t: wide_gradients(t, False), heads)
            if dtype == torch.float32:
                # (For float16 tensors that are not aligned PyTorch itself has no kernel on the
                # GPU: "cutlassF: no kernel found to launch!", whatever the GPU.)
                case('attention.grad.%d.unaligned.%s' % (width, tag), lambda t: wide_gradients(t, True), heads)
    return cases


def setitem(x, where, value):
    x = x.clone()
    x[where] = value
    return x


def on_cpu(result):
    """A tensor, or tensors within tuples (of names or not) within tuples, brought back."""
    return [on_cpu(t) for t in result] if isinstance(result, (tuple, list)) else result.cpu()


def stats(t):
    """Of random numbers: what must agree between two generators, roughly (mean, deviation, and for
    numbers within bounds the least and the greatest)."""
    import torch
    return torch.stack([t.mean(), t.std(), t.min(), t.max()]).round(decimals=1)


def several(result):
    """Whether a result is several (a tuple, of names or not, or a list), not one tensor."""
    return isinstance(result, (tuple, list)) or hasattr(result, '_fields')


def shapes(a, b):
    """How the results a (the CPU's) and b (the GPU's) differ in how they are made: several or
    one, how many, a tensor or not; None when alike. Compared before any arithmetic, which would
    broadcast one shape to the other, or `zip`, which stops at the shorter."""
    import torch
    if several(a) or several(b):
        if not (several(a) and several(b)):
            return 'a result of %d on the CPU, %d on the GPU' % (len(a) if several(a) else 1, len(b) if several(b) else 1)
        if len(a) != len(b):
            return 'another number of results: %d on the CPU, %d on the GPU' % (len(a), len(b))
        return None
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        return 'a %s on the CPU, a %s on the GPU' % (type(a).__name__, type(b).__name__)
    return None


def same(a, b, name):
    """None when the two results agree, else how they differ."""
    import torch
    different = shapes(a, b)
    if different:
        return different
    if several(a):
        return next((d for d in (same(x, y, name) for x, y in zip(a, b)) if d), None)
    if a.shape != b.shape or a.dtype != b.dtype:
        return 'shape or type: %s %s on the CPU, %s %s on the GPU' % (tuple(a.shape), a.dtype, tuple(b.shape), b.dtype)
    if not (a.dtype.is_floating_point or a.dtype.is_complex):
        wrong = (a != b).sum().item()
        return '%d of %d elements' % (wrong, a.numel()) if wrong else None
    if name.startswith(('rand', 'randn', 'randint')):
        # Two generators: the moments, to one decimal, give or take.
        return None if (a - b).abs().max().item() <= 0.11 else 'moments %s and %s' % (a.tolist(), b.tolist())
    rtol, atol = {torch.float16: (2e-2, 2e-2), torch.float32: (2e-4, 2e-5)}.get(a.dtype, (1e-9, 1e-11))
    if name == 'hardswish.f64':
        rtol = 1e-7                     # PyTorch's kernel for the GPU divides by a six held as a float
    a64, b64 = a.double(), b.double()
    close = torch.isclose(a64, b64, rtol=rtol, atol=atol, equal_nan=True)
    if close.all():
        return None
    worst = ((a64 - b64).abs() / (a64.abs() + atol))[~close].nan_to_num(float('inf')).max().item()
    return '%d of %d elements, the worst off by %.3g of its value' % ((~close).sum().item(), a.numel(), worst)


def through_double(expected, got, truth):
    """For a float32 result that is a sum of many products, where a tolerance by element says
    nothing (the error of an element goes with the largest term of its sum, not with its own
    size: gradients through attention): the same computation in float64 on the CPU is the
    truth, and the GPU's float32 must be as near it as the CPU's float32 is, within a factor.
    Measured (heads of 64 to 256): the two are off the truth by 2.5e-5 and 2.7e-5 at most for
    heads of 64, 9.5e-5 and 9.0e-5 for heads of 256, and by the same on average (1.4e-6 and
    1.4e-6; 5.32e-6 and 5.33e-6), while they differ from each other by more than any tolerance
    by element allows at elements near zero. None when they agree, else how they differ."""
    different = shapes(expected, got) or shapes(expected, truth)
    if different:
        return different
    if several(expected):
        return next((d for d in (through_double(a, b, t) for a, b, t in zip(expected, got, truth)) if d), None)
    if expected.shape != got.shape or expected.dtype != got.dtype:
        return 'shape or type: %s %s on the CPU, %s %s on the GPU' % (tuple(expected.shape), expected.dtype, tuple(got.shape), got.dtype)
    cpu, gpu = (expected.double() - truth).abs(), (got.double() - truth).abs()
    worst, mean = (cpu.max().item(), gpu.max().item()), (cpu.mean().item(), gpu.mean().item())
    if gpu.isnan().any() or worst[1] > 3 * worst[0] or mean[1] > 2 * mean[0]:
        return 'off the float64 result by %.3g at most and %.3g on average, where the CPU in float32 is off by %.3g and %.3g' % (worst[1], mean[1], worst[0], mean[0])
    return None


def comparison_checks():
    """What same and through_double must refuse, and take, on tensors of the CPU: a check that was
    never seen to fail has not been shown to check. (Until 2026-11-23 a tuple of two results was
    compared row by row with a tensor of two rows, and through_double compared the gradients
    that both had: a missing last one passed.)"""
    import torch
    one, rows = torch.ones(3), torch.ones(2, 3)
    truth = one.double()
    wrong = []
    for what, difference, ok, said in [
        ('the same', same((one, rows), [one, rows], 'x'), True, None),
        ('a tuple and a tensor of as many rows', same((one, one), rows, 'x'), False, 'a result of 2 on the CPU, 1 on the GPU'),
        ('a tensor and a tuple', same(rows, (one, one), 'x'), False, 'a result of 1 on the CPU, 2 on the GPU'),
        ('one result fewer', same((one, one), [one], 'x'), False, 'another number of results: 2 on the CPU, 1 on the GPU'),
        ('a row where the CPU has rows', same(rows, one, 'x'), False, 'shape or type: (2, 3)'),
        ('float64 where the CPU has float32', same(one, truth, 'x'), False, 'shape or type: (3,) torch.float32 on the CPU, (3,) torch.float64'),
        ('gradients, the same', through_double((one, one), [one, one], (truth, truth)), True, None),
        ('gradients, the last one missing', through_double((one, one), [one], (truth, truth)), False, 'another number of results: 2 on the CPU, 1 on the GPU'),
        ('gradients, one where the CPU has two', through_double((one, one), one, (truth, truth)), False, 'a result of 2 on the CPU, 1 on the GPU'),
        ('a gradient of another shape', through_double(one, rows, truth), False, 'shape or type'),
    ]:
        if (difference is None) != ok or (said and said not in (difference or '')):
            wrong.append('%s: %r (should %s%s)' % (what, difference, 'pass' if ok else 'fail', ', saying %r' % said if said else ''))
    return wrong


def worker(first, channel, patterns):
    """Run the cases from number `first` on. On the descriptor `channel`: the number of cases
    selected ('cases', N), then a line when a case starts and one when it ends: number, name,
    verdict, detail."""
    import faulthandler
    import torch
    os.set_inheritable(channel, False)
    out = os.fdopen(channel, 'w', buffering=1)

    def say(*fields):
        out.write('\t'.join(str(f) for f in fields) + '\n')
    # (Where it was, on standard error, when it is stopped for taking too long.)
    faulthandler.register(signal.SIGUSR1)
    cases = build()
    say('cases', sum(1 for name, fn, inputs in cases if not patterns or any(p in name for p in patterns)))
    torch.zeros(1, device='cuda')
    for number, (name, fn, inputs) in enumerate(cases):
        if number < first or (patterns and not any(p in name for p in patterns)):
            continue
        say(number, name, 'start', '')
        try:
            expected = fn('cpu') if inputs == (None,) else fn(*inputs)
        except Exception as e:
            say(number, name, 'skipped', str(e).split('\n')[0][:150])
            continue
        try:
            got = fn('cuda') if inputs == (None,) else fn(*[t.cuda() for t in inputs])
            torch.cuda.synchronize()
            got = on_cpu(got)
            if name.startswith('attention.grad') and name.endswith('.f32'):
                difference = through_double(expected, got, fn(*[t.double() for t in inputs]))
            else:
                difference = same(expected, got, name)
            verdict, detail = ('DIFFERENT', difference) if difference else ('ok', '')
        except Exception as e:
            verdict, detail = 'error', '%s: %s' % (type(e).__name__, str(e).split('\n')[0][:200])
            # A kernel refused at the lookup comes back as one of these two, and leaves the process as
            # it was. After a fault nothing works any more: the next cases need a new process.
            if 'PTX JIT compilation failed' in detail or 'invalid device function' in detail:
                verdict, detail = 'refused', ''
            try:
                (torch.zeros(4, device='cuda') + 1).cpu()
                torch.cuda.synchronize()
            except Exception:
                say(number, name, 'CRASH', detail)
                out.flush()
                os._exit(3)
        say(number, name, verdict, detail)


def lines(fd, wait):
    """What a worker writes on its channel, line by line, until it closes it; None as the last
    item if it writes nothing for `wait` seconds."""
    pending = b''
    while True:
        if not select.select([fd], [], [], wait)[0]:
            yield None
            return
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        pending += chunk
        *complete, pending = pending.split(b'\n')
        for line in complete:
            yield line.decode(errors='replace')
    if pending:
        yield pending.decode(errors='replace')


def stop(p):
    """Stop a worker: SIGUSR1 first (its fault handler then writes where it was on its standard
    error), then SIGKILL. False if it has not ended even so."""
    for sig, wait in ((signal.SIGUSR1, 1), (signal.SIGKILL, 60)):
        p.send_signal(sig)
        try:
            p.wait(wait)
            return True
        except subprocess.TimeoutExpired:
            pass
    return False


def tail(path, offset):
    """What was written to the file `path` from `offset` on: its last 2,000 characters at most."""
    with open(path, 'rb') as f:
        f.seek(max(offset, os.path.getsize(path) - 2000))
        return f.read().decode(errors='replace').strip()


def main():
    args = sys.argv[1:]
    if args[:1] == ['--worker']:
        return worker(int(args[1]), int(args[2]), args[3:])
    if args[:1] == ['--comparison']:
        wrong = comparison_checks()
        for line in wrong:
            print('WRONG  the comparison: ' + line)
        print('FAIL  the comparison' if wrong else 'PASS  the comparison')
        return 1 if wrong else 0
    verbose = '-v' in args
    patterns = [a for a in args if a != '-v']
    wait = int(os.environ.get('VF_CASE_WAIT', '600'))
    # (With the other logs when the script that runs this says where they are: not in /tmp,
    # which on the nodes is memory.)
    logs = os.path.join(os.environ['VF_OUT'], 'logs') if os.environ.get('VF_OUT') else None
    if logs:
        os.makedirs(logs, exist_ok=True)
    log = tempfile.NamedTemporaryFile(prefix='vf-torch-', suffix='.log', delete=False, dir=logs).name
    output, errors = log[:-len('.log')] + '.out', log[:-len('.log')] + '.err'
    env = dict(os.environ, VF_LOG=log, VF_LOG_LEVEL=os.environ.get('VF_LOG_LEVEL', '1'))
    results, first, expected, problems = {}, 0, None, []

    def problem(text):
        problems.append(text)
        print('%-9s %s' % ('RUN', text), flush=True)
    while True:
        offset = os.path.getsize(errors) if os.path.exists(errors) else 0
        r, w = os.pipe()
        with open(output, 'ab') as out, open(errors, 'ab') as err:
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), '--worker', str(first), str(w)] + patterns,
                                 env=env, stdout=out, stderr=err, pass_fds=(w,))
        os.close(w)
        running, last, silent = None, None, False
        for line in lines(r, wait):
            if line is None:
                silent = True
                break
            fields = line.split('\t', 3)
            if len(fields) == 2 and fields[0] == 'cases' and fields[1].isdigit():
                if expected is not None and int(fields[1]) != expected:
                    problem('a worker said %s cases, the first %d' % (fields[1], expected))
                expected = int(fields[1])
                continue
            if len(fields) != 4 or not fields[0].isdigit() or fields[2] not in VERDICTS + ('start',):
                problem('a worker wrote what is not a verdict: %r' % line[:200])
                continue
            number, name, verdict, detail = int(fields[0]), fields[1], fields[2], fields[3]
            if verdict == 'start':
                running = (number, name)
                continue
            results[number] = (name, verdict, detail)
            running, last = None, verdict
            if verdict == 'CRASH':
                first = number + 1
            if verbose or verdict not in ('ok', 'skipped', 'refused'):
                print('%-9s %-28s %s' % (verdict, name, detail), flush=True)
        os.close(r)
        # (The worker closes its channel when it ends; it may also be stuck after that.)
        ended = not silent
        if ended:
            try:
                p.wait(wait)
            except subprocess.TimeoutExpired:
                ended = False
        if not ended and not stop(p):
            problem('the worker from case %d did not end even with SIGKILL: the run ends here' % first)
            break
        if running is not None:
            # The worker ended inside a case, or was stopped there.
            detail = ('no end within %d s: stopped' % wait if silent
                      else 'exit status %d; %s' % (p.returncode, tail(errors, offset).split('\n')[-1][:200]))
            results[running[0]] = (running[1], 'CRASH', detail)
            print('%-9s %-28s %s' % ('CRASH', running[1], detail), flush=True)
            first = running[0] + 1
            continue
        if not ended:
            problem('the worker from case %d %s and was stopped' % (
                first, 'wrote nothing for %d s outside a case' % wait if silent else 'did not end within %d s after its last case' % wait))
            break
        if p.returncode == 3 and last == 'CRASH':
            continue                    # it lost the GPU in a case and said so: on from the next
        if p.returncode != 0:
            if not results:
                print(tail(errors, offset))
                return 2
            problem('the worker from case %d ended with status %d outside a case: %s' % (
                first, p.returncode, tail(errors, offset).split('\n')[-1][:200]))
        break
    if expected is None:
        print(tail(errors, 0))
        print('no worker said how many cases there are')
        return 2
    if len(results) != expected:
        problem('%d verdicts for %d cases' % (len(results), expected))
    count = {}
    for name, verdict, detail in results.values():
        count[verdict] = count.get(verdict, 0) + 1
    print('%d cases: %s' % (len(results), ', '.join('%d %s' % (n, v) for v, n in sorted(count.items(), key=lambda kv: -kv[1]))))
    if problems:
        print('the run itself went wrong %d time%s (the RUN lines)' % (len(problems), '' if len(problems) == 1 else 's'))
    reasons = {}
    for line in open(log, errors='replace') if os.path.exists(log) else []:
        m = re.search(r'is not given to the program: (.*)$', line)
        if m:
            reason = re.sub(r'0x[0-9a-f]+', 'N', m.group(1))
            reasons[reason] = reasons.get(reason, 0) + 1
    skipped = {}
    for name, verdict, detail in results.values():
        if verdict == 'skipped':
            skipped.setdefault(detail[:110], []).append(name)
    if skipped:
        # (So that a case that tests nothing because of a mistake of its own is seen: the sorts
        # with `stable` were skipped for months for an argument PyTorch rejects on the CPU too.)
        print('skipped (the CPU did not do it), by what it said:')
        for detail, names in sorted(skipped.items(), key=lambda kv: -len(kv[1])):
            print('  %4d  %s   (%s%s)' % (len(names), detail, ', '.join(names[:3]), ', ...' if len(names) > 3 else ''))
    if reasons:
        print('kernels refused, by reason:')
        for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print('  %4d  %s' % (n, reason[:160]))
    print('the proxy log: %s' % log)
    print('what the workers printed: %s, %s' % (output, errors))
    return 1 if count.get('DIFFERENT') or count.get('CRASH') or problems else 0


if __name__ == '__main__':
    sys.exit(main())
