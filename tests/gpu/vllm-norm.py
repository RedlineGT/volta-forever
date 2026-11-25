#!/usr/bin/env python3
"""vLLM's fused_add_rms_norm kernel on the GPU against the same computation on the CPU.

The operation that vLLM's engine runs at every layer when its compilation is off: `residual =
x + residual`, then `x = residual / sqrt(mean(residual^2) + eps) * weight`, each rounded once to
the type of the data. For bfloat16 and float16 data of several shapes (rows of 896, 4096 and 1000
elements, which take the kernel's path of eight elements at a time, and of 8), with values of
three sizes, with a weight and without one (two kernels; NORM_ONLY=weighted or unweighted in the
environment runs the cases of one):
  - the residual must be the CPU's, bit for bit (a sum of two values, rounded once);
  - the norm must be within one unit in the last place of the CPU's for every element (the kernel
    adds the squares in another order than the CPU and takes the reciprocal square root from the
    GPU's own instruction).
Prints one line for each case, then `verdict: NORM_MATCH` or `verdict: NORM_DIFFER`, then PASS, or
FAIL with the error (a kernel that the program is not given: the layer says which on standard
error); the exit status is 1 after FAIL. Told 8.0, the bfloat16 kernel is one that only the sm_80
code of vLLM 0.34's library has (LIMITS.md, "The capability told with VF_CAPABILITY").
"""
import os, sys, traceback
import torch


def bits(t):
    return t.contiguous().view(torch.int16).to(torch.int32) & 0xffff


def main():
    import vllm._custom_ops as ops
    print('told', torch.cuda.get_device_capability(), flush=True)
    torch.manual_seed(0)
    good = True
    only = os.environ.get('NORM_ONLY')
    for dtype in (torch.bfloat16, torch.float16):
        # (With a weight and without one: two kernels of the library, each with its own name.)
        for rows, hidden, scale, weighted in ((64, 896, 1.0, True), (7, 896, 30.0, True), (3, 4096, 0.01, True),
                                              (5, 1000, 1.0, True), (2, 8, 1.0, True),
                                              (64, 896, 1.0, False), (3, 4096, 30.0, False), (2, 8, 0.01, False)):
            if only and only != ('weighted' if weighted else 'unweighted'):
                continue
            x = (torch.randn(rows, hidden) * scale).to(dtype)
            r = (torch.randn(rows, hidden) * scale).to(dtype)
            w = (torch.rand(hidden) + 0.5).to(dtype) if weighted else torch.ones(hidden, dtype=dtype)
            s = (x.float() + r.float()).to(dtype)
            var = s.float().pow(2).mean(-1, keepdim=True)
            want = (s.float() * torch.rsqrt(var + 1e-6) * w.float()).to(dtype)
            gx, gr, gw = x.cuda(), r.cuda(), w.cuda() if weighted else None
            ops.fused_add_rms_norm(gx, gr, gw, 1e-6)
            torch.cuda.synchronize()
            got, residual = gx.cpu(), gr.cpu()
            wrong = int((bits(residual) != bits(s)).sum())
            away = (bits(got) - bits(want)).abs()
            good &= wrong == 0 and int(away.max()) <= 1
            print(f'  {str(dtype)[6:]} {rows}x{hidden}, values of about {scale}{"" if weighted else ", no weight"}: residual {wrong} of {s.numel()} differ '
                  f'from the CPU\'s bits; norm {int((away == 0).sum())} the CPU\'s bits, {int((away == 1).sum())} one '
                  f'unit away, {int((away > 1).sum())} further (largest {int(away.max())})', flush=True)
    print('verdict:', 'NORM_MATCH' if good else 'NORM_DIFFER')
    print('PASS')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f'FAIL: {e}')
        traceback.print_exc()
        sys.exit(1)
