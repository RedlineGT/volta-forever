"""Triton's matrix products with VF_TRITON_TC on and off, on a V100 (tests/gpu/triton-tc.sh runs it).

  python3 triton-tc.py probe | kernels [--vllm] | compile

One JSON line per case. Every output is compared with a computation in float64 on the CPU; the
line says what Triton made of the kernel (the target of its PTX), the error, the time and a digest
of the output. Which mode is measured is the environment's (VF_TRITON_TC, through the launcher).
"""
import hashlib
import json
import math
import os
import re
import sys
import time

import torch

R = {}


def emit(**kw):
    print(json.dumps(dict(R, **kw)), flush=True)


def timeit(fn, reps=10):
    """Microseconds a launch: the fastest of 5 batches of `reps` launches (a short kernel's batch
    is as long as Python takes to launch it when the machine is busy: the fastest batch is the
    kernel's)."""
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps):
            fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / reps * 1e3)
    return min(ts)


def err(out, ref):
    out, ref = out.double().cpu(), ref.double().cpu()
    return float((out - ref).abs().max() / ref.abs().max().clamp_min(1e-30))


def digest(t):
    return hashlib.sha256(t.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def target(kernel):
    try:
        return re.search(r'\.target (\S+)', kernel.asm['ptx']).group(1)
    except Exception:  # noqa: BLE001
        return '?'


def probe():
    import triton
    from triton.runtime import driver
    rec = dict(torch='%d.%d' % torch.cuda.get_device_capability(), triton=driver.active.get_current_target().arch)
    try:
        import pynvml
        pynvml.nvmlInit()
        rec['nvml'] = '%d.%d' % pynvml.nvmlDeviceGetCudaComputeCapability(pynvml.nvmlDeviceGetHandleByIndex(0))
    except Exception as e:  # noqa: BLE001
        rec['nvml'] = 'n/a: ' + str(e)[:60]
    rec['version'] = triton.__version__
    emit(what='probe', **rec)


def kernels(with_vllm):
    import triton
    import triton.language as tl

    @triton.jit
    def mm(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        om = (pm * BM + tl.arange(0, BM)) % M
        on = (pn * BN + tl.arange(0, BN)) % N
        ok = tl.arange(0, BK)
        a_ptrs = a_ptr + om[:, None] * sam + ok[None, :] * sak
        b_ptrs = b_ptr + ok[:, None] * sbk + on[None, :] * sbn
        acc = tl.zeros([BM, BN], dtype=c_ptr.dtype.element_ty)
        for k in range(0, tl.cdiv(K, BK)):
            acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc, out_dtype=c_ptr.dtype.element_ty)
            a_ptrs += BK * sak
            b_ptrs += BK * sbk
        cm = pm * BM + tl.arange(0, BM)
        cn = pn * BN + tl.arange(0, BN)
        tl.store(c_ptr + cm[:, None] * scm + cn[None, :] * scn, acc, mask=(cm[:, None] < M) & (cn[None, :] < N))

    cases = [
        # (name, operand type, M, N, K, BM, BN, BK, warps, stages)
        ('f16', torch.float16, 2048, 4096, 1536, 128, 128, 32, 8, 1),
        ('f16-s3', torch.float16, 2048, 4096, 1536, 64, 64, 32, 4, 3),
        ('f16-decode', torch.float16, 16, 7344, 4096, 16, 64, 64, 4, 2),
        ('f16-shared', torch.float16, 2048, 4096, 1536, 128, 256, 64, 8, 4),
        ('bf16', torch.bfloat16, 2048, 4096, 1536, 64, 64, 32, 4, 3),
        ('i8', torch.int8, 1024, 2048, 2048, 64, 64, 64, 4, 2),
        ('f32', torch.float32, 1024, 2048, 1024, 64, 64, 32, 4, 3),
    ]
    g = torch.Generator().manual_seed(0)
    for name, dt, M, N, K, bm, bn, bk, w, s in cases:
        if dt == torch.int8:
            a = torch.randint(-128, 128, (M, K), generator=g, dtype=dt)
            b = torch.randint(-128, 128, (K, N), generator=g, dtype=dt)
            c = torch.empty(M, N, device='cuda', dtype=torch.int32)
        else:
            a = (torch.randn(M, K, generator=g) * 0.5).to(dt)
            b = (torch.randn(K, N, generator=g) * 0.5).to(dt)
            c = torch.empty(M, N, device='cuda', dtype=torch.float32)
        ad, bd = a.cuda(), b.cuda()
        rows = min(M, 256)
        ref = a[:rows].double() @ b.double()
        grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))

        def run():
            return mm[grid](ad, bd, c, M, N, K, ad.stride(0), ad.stride(1), bd.stride(0), bd.stride(1), c.stride(0),
                            c.stride(1), BM=bm, BN=bn, BK=bk, num_warps=w, num_stages=s)
        rec = dict(what='kernel', case=name, shape=f'{M}x{N}x{K}', cfg=f'{bm}x{bn}x{bk} w{w} s{s}')
        try:
            k = run()
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            emit(**rec, error=f'{type(e).__name__}: {str(e)[:300]}')
            continue
        e = 0.0 if dt == torch.int8 and bool((c[:rows].cpu().long() == ref.long()).all()) else err(c[:rows], ref)
        emit(**rec, target=target(k), shared=k.metadata.shared, err=e, hash=digest(c), us=round(timeit(run), 1))
    if with_vllm:
        vllm_kernels()


def moe_align(topk_ids, num_experts, block_m):
    import triton
    flat = topk_ids.reshape(-1).to(torch.int64)
    n = flat.numel()
    max_padded = n + num_experts * (block_m - 1)
    sorted_ids = torch.full((max_padded,), n, dtype=torch.int32)
    expert_ids = torch.full((triton.cdiv(max_padded, block_m),), -1, dtype=torch.int32)
    pos = 0
    for e in range(num_experts):
        idx = (flat == e).nonzero().reshape(-1)
        if idx.numel() == 0:
            continue
        padded = triton.cdiv(idx.numel(), block_m) * block_m
        sorted_ids[pos:pos + idx.numel()] = idx.to(torch.int32)
        expert_ids[pos // block_m:(pos + padded) // block_m] = e
        pos += padded
    return sorted_ids, expert_ids, torch.tensor([pos], dtype=torch.int32)


def vllm_kernels():
    """vLLM's fused_moe_kernel (its default configuration) and kernel_unified_attention."""
    import triton
    import triton.language as tl
    from vllm.model_executor.layers.fused_moe import fused_moe as fm
    from vllm.v1.attention.ops import triton_unified_attention as ua
    E, topk, K, N = 64, 8, 2048, 1024
    g = torch.Generator().manual_seed(1)
    for dt in (torch.float16, torch.bfloat16):
        ct = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}[dt]
        b = (torch.randn(E, N, K, generator=g) * 0.05).to(dt)
        bd = b.cuda()
        for M in (16, 512):
            a = torch.randn(M, K, generator=g).to(dt)
            ids = torch.stack([torch.randperm(E, generator=g)[:topk] for _ in range(M)]).to(torch.int32)
            cfg = fm.get_default_config(M, E, N, K, topk, None)
            s, eids, npad = moe_align(ids, E, cfg['BLOCK_SIZE_M'])
            ad, sd, ed, nd = a.cuda(), s.cuda(), eids.cuda(), npad.cuda()
            c = torch.zeros(M, topk, N, device='cuda', dtype=dt)

            def run():
                fm.invoke_fused_moe_triton_kernel(ad, bd, c, None, None, None, sd, ed, nd, False, topk, cfg, ct,
                                                  False, False, False, False, False, None)
            rec = dict(what='kernel', case='vllm-fused-moe-%s-%d' % (str(dt)[6:], M))
            try:
                run()
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                emit(**rec, error=f'{type(e).__name__}: {str(e)[:300]}')
                continue
            m = min(M, 32)
            ref = torch.zeros(m, topk, N, dtype=torch.float64)
            for e in ids[:m].unique().tolist():
                t_, j_ = (ids[:m] == e).nonzero(as_tuple=True)
                ref[t_, j_] = a[t_].double() @ b[e].double().t()
            emit(**rec, err=err(c[:m], ref), hash=digest(c), us=round(timeit(run, 3), 1))
    Hq, Hkv, D, page = 16, 4, 128, 16
    for dt in (torch.float16, torch.bfloat16):
        for kind, ql, kl in (('prefill', [512] * 2, [512] * 2), ('decode', [1] * 32, [1024] * 32)):
            nblk = [triton.cdiv(n, page) for n in kl]
            total = sum(nblk) + 3
            perm = torch.randperm(total, generator=g).to(torch.int32)
            bt = torch.zeros(len(kl), max(nblk), dtype=torch.int32)
            p = 0
            for i, n in enumerate(nblk):
                bt[i, :n] = perm[p:p + n]
                p += n
            T = sum(ql)
            q = torch.randn(T, Hq, D, generator=g).to(dt)
            k = torch.randn(total, page, Hkv, D, generator=g).to(dt)
            v = torch.randn(total, page, Hkv, D, generator=g).to(dt)
            cu = torch.tensor([0] + [sum(ql[:i + 1]) for i in range(len(ql))], dtype=torch.int32)
            qd, kd, vd, cud = q.cuda(), k.cuda(), v.cuda(), cu.cuda()
            kld, btd = torch.tensor(kl, dtype=torch.int32).cuda(), bt.cuda()
            o = torch.zeros(T, Hq, D, device='cuda', dtype=dt)
            scale = 1.0 / math.sqrt(D)

            def run():
                ua.unified_attention(qd, kd, vd, o, cud, max(ql), kld, max(kl), scale, True, (-1, -1), btd, 0,
                                     None, None, None)
            rec = dict(what='kernel', case='vllm-attention-%s-%s' % (str(dt)[6:], kind))
            try:
                run()
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                emit(**rec, error=f'{type(e).__name__}: {str(e)[:300]}')
                continue
            errs = []
            for i in range(2):
                pages = bt[i, :nblk[i]].long()
                ki = k[pages].reshape(-1, Hkv, D)[:kl[i]].double()
                vi = v[pages].reshape(-1, Hkv, D)[:kl[i]].double()
                qi = q[cu[i]:cu[i + 1]].double()
                for h in (0, Hq - 1):
                    kv = h // (Hq // Hkv)
                    sc = (qi[:, h] @ ki[:, kv].t()) * scale
                    pos = torch.arange(kl[i] - ql[i], kl[i]).unsqueeze(1)
                    sc = sc.masked_fill(torch.arange(kl[i]).unsqueeze(0) > pos, float('-inf'))
                    errs.append(err(o[cu[i]:cu[i + 1], h], torch.softmax(sc, dim=-1) @ vi[:, kv]))
            emit(**rec, err=max(errs), hash=digest(o), us=round(timeit(run, 3), 1))


def compile_mm():
    """torch.compile of a matrix product, with Inductor's Triton template as the only choice."""
    import torch._inductor.config as ic
    ic.max_autotune = True
    ic.max_autotune_gemm_backends = 'TRITON'
    ic.autotune_fallback_to_aten = False
    g = torch.Generator().manual_seed(3)
    a = (torch.randn(1024, 2048, generator=g) * 0.5).half()
    b = (torch.randn(2048, 1536, generator=g) * 0.5).half()
    ad, bd = a.cuda(), b.cuda()
    f = torch.compile(lambda x, y: torch.relu(x @ y), mode='max-autotune-no-cudagraphs')
    t0 = time.time()
    try:
        out = f(ad, bd)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        emit(what='compile', error=f'{type(e).__name__}: {str(e)[:400]}')
        return
    seconds = time.time() - t0
    ref = torch.relu(a.double() @ b.double())
    emit(what='compile', err=err(out, ref), hash=digest(out), us=round(timeit(lambda: f(ad, bd)), 1),
         compile_seconds=round(seconds, 1))


def main():
    R.update(tc=os.environ.get('VF_TRITON_TC', ''), label=os.environ.get('VF_TC_LABEL', ''))
    what = sys.argv[1]
    if what == 'probe':
        probe()
    elif what == 'kernels':
        kernels('--vllm' in sys.argv)
    elif what == 'compile':
        compile_mm()


if __name__ == '__main__':
    main()
