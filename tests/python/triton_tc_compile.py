"""Compile one Triton kernel for a target, with no GPU, and print what Triton made (one JSON line).

  python3 triton_tc_compile.py KERNEL [--arch 75] [--block BMxBNxBK] [--warps 8] [--stages 3] [--inproc]

KERNEL: mm-f16, mm-bf16, mm-i8, mm-f32 (a tl.dot of two operands of that type in a loop over K),
mmt-* (the same with B stored N x K), add (no tl.dot). Used by tests/python/triton_tc.py with
VF_TRITON_TC on and off: the line says the target of the PTX, how many mma instructions it has,
whether the cubin keeps its PTX, the shared memory, the SHA-256 of the cubin and the directories of
Triton's cache (TRITON_CACHE_DIR; not the layer's, volta-forever-UID). With --inproc (and the switch
off), the kernel's TTIR for 7.5 is compiled for sm_80 in this process by the helper's compile80
(python/volta_forever_triton_helper.py), as a helper compiles it.
"""
import argparse
import hashlib
import json
import os
import re

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def mm(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    a_ptrs = a_ptr + om[:, None] * K + ok[None, :]
    b_ptrs = b_ptr + ok[:, None] * N + on[None, :]
    acc = tl.zeros([BM, BN], dtype=c_ptr.dtype.element_ty)
    for k in range(0, tl.cdiv(K, BK)):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc, out_dtype=c_ptr.dtype.element_ty)
        a_ptrs += BK
        b_ptrs += BK * N
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc)


@triton.jit
def mmt(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """mm with B stored N x K (its elements of one n next to each other)."""
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    a_ptrs = a_ptr + om[:, None] * K + ok[None, :]
    b_ptrs = b_ptr + on[None, :] * K + ok[:, None]
    acc = tl.zeros([BM, BN], dtype=c_ptr.dtype.element_ty)
    for k in range(0, tl.cdiv(K, BK)):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc, out_dtype=c_ptr.dtype.element_ty)
        a_ptrs += BK
        b_ptrs += BK
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc)


@triton.jit
def add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(o_ptr + i, tl.load(x_ptr + i, mask=m) + tl.load(y_ptr + i, mask=m), mask=m)


def source(fn, signature, constants):
    """The source of a kernel whose pointers and integers are multiples of 16 (as the JIT finds
    them for most launches), with the parameters of ASTSource of Triton 3.3 to 3.9."""
    attrs = {(i,): [["tt.divisibility", 16]] for i, (k, v) in enumerate(signature.items()) if v != 'constexpr'}
    try:
        return ASTSource(fn, signature, constexprs=constants, attrs=attrs)
    except TypeError:
        return ASTSource(fn, signature, constants=constants)


def inproc(src, a):
    """(PTX, cubin) of the helper's compile80 in this process, of the TTIR, metadata and options that
    Triton has for 7.5 when its stage ttgir starts (where the hook would ask a helper), in a cache
    of its own (the compilation for 7.5 stops there)."""
    import importlib.util
    import tempfile
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python', 'volta_forever_triton_helper.py')
    spec = importlib.util.spec_from_file_location('vft_helper_inproc', here)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    from triton.backends.nvidia.compiler import CUDABackend
    from triton._C.libtriton import ir
    seen = {}

    class Stop(Exception):
        pass

    add_stages = CUDABackend.add_stages

    def capture(self, stages, options, *rest):
        add_stages(self, stages, options, *rest)

        def ttgir(mod, metadata):
            seen.update(backend=self, options=options, ttir=str(mod), metadata=dict(metadata))
            raise Stop()
        stages['ttgir'] = ttgir

    CUDABackend.add_stages = capture
    with tempfile.TemporaryDirectory() as cache:
        os.environ['TRITON_CACHE_DIR'] = cache
        try:
            triton.compile(src, target=GPUTarget('cuda', 75, 32), options=dict(num_warps=a.warps, num_stages=a.stages))
        except Stop:
            pass
        finally:
            CUDABackend.add_stages = add_stages
    context = ir.context()
    ir.load_dialects(context)
    seen['backend'].load_dialects(context)
    result, why, _ = helper.compile80(seen['backend'], seen['ttir'], context, seen['metadata'], seen['options'])
    if result is None:
        raise SystemExit('compile80: %s' % why)
    return result['ptx'], result['cubin']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kernel')
    ap.add_argument('--arch', type=int, default=75)
    ap.add_argument('--block', default='64x64x32')
    ap.add_argument('--warps', type=int, default=4)
    ap.add_argument('--stages', type=int, default=3)
    ap.add_argument('--inproc', action='store_true')
    a = ap.parse_args()
    if a.kernel == 'add':
        src = source(add, {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32', 'n': 'i32', 'BLOCK': 'constexpr'},
                     {'BLOCK': 1024})
    else:
        t = a.kernel.split('-')[1]
        operand = {'f16': '*fp16', 'bf16': '*bf16', 'i8': '*i8', 'f32': '*fp32'}[t]
        result = '*i32' if t == 'i8' else '*fp32'
        bm, bn, bk = (int(v) for v in a.block.split('x'))
        src = source(mmt if a.kernel.startswith('mmt-') else mm,
                     {'a_ptr': operand, 'b_ptr': operand, 'c_ptr': result, 'M': 'i32', 'N': 'i32', 'K': 'i32',
                      'BM': 'constexpr', 'BN': 'constexpr', 'BK': 'constexpr'}, {'BM': bm, 'BN': bn, 'BK': bk})
    if a.inproc:
        ptx, cubin = inproc(src, a)
        print(json.dumps(dict(triton=triton.__version__, target=re.search(r'\.target (\S+)', ptx).group(1),
                              cubin=hashlib.sha256(cubin).hexdigest()[:16])))
        return
    k = triton.compile(src, target=GPUTarget('cuda', a.arch, 32), options=dict(num_warps=a.warps, num_stages=a.stages))
    ptx = k.asm['ptx']
    cubin = k.asm['cubin']
    cache = os.environ.get('TRITON_CACHE_DIR', '')
    print(json.dumps(dict(
        triton=triton.__version__,
        target=re.search(r'\.target (\S+)', ptx).group(1),
        mma=len(re.findall(r'\bmma\.sync', ptx)),
        keeps_ptx=b'.nv_debug_ptx_txt' in cubin,
        shared=k.metadata.shared,
        num_stages=k.metadata.num_stages,
        cubin=hashlib.sha256(cubin).hexdigest()[:16],
        cache=sorted(d for d in os.listdir(cache) if not d.startswith('volta-forever-'))
        if cache and os.path.isdir(cache) else [])))


if __name__ == '__main__':
    main()
