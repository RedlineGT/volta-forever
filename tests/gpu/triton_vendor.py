"""The Triton kernels that vLLM and SGLang ship, as cases for tests/gpu/triton-diff.py (--vendor).

`cases(level)` gives the vLLM cases when `import vllm` works and the SGLang cases when `import
sglang` works, so that one file runs in either environment (each package has its own Python: the
two are not installed together). The module is imported both on the GPU and under Triton's
interpreter (TRITON_INTERPRET=1, no GPU); it must not touch a GPU at import, and the packages
are imported only when `cases` is called.

Each case launches a kernel of the package, with the arguments its engine gives it on the hot
path (shapes of small and 7B-70B models) and adversarial ones (lengths that are not a multiple of
the block, one token, rows that end at a block boundary, the largest values). Most cases call the
package's own launcher (the Python function that computes the grid and the meta-parameters and
launches the kernel), with tensors on the GPU or, for the reference, on the CPU under the
interpreter: one kernel, one launch, on two machines. Where a launcher does something the
interpreter cannot (a CUDA extension, a query of the device), the case launches the kernel itself
with the launcher's arguments.

Autotuned kernels are pinned to one of their configurations (`pin`): an autotuner chooses by
timing, which would make the GPU runs choose different code, and the interpreter cannot time.

The interpreter cannot run a few kernels as written; for these the reference runs a copy of the
module whose source has the smallest change that lets it run (`patched`), named in the case's note:
  - `x.to(tl.int64)` where x is the variable of a `range` loop: the interpreter gives it as a
    Python int, which has no `.to`.
Where the interpreter computes something else than the GPU (bfloat16 arithmetic: it rounds a
conversion to bfloat16 toward zero and does not compute products of bfloat16 values), the
reference is the interpreter on float32 copies (`interp32`, see triton_corpus.Case) or a
computation with PyTorch on the CPU, said in the case.

Data: small integers as floats wherever the result of the kernel is then one value whatever the
order of its sums (matrix products, MoE, LoRA, scans of sums, gathers and copies): compared bit for
bit. Softmax, exponentials, normalisations and random data: compared within a tolerance tied to the
operation and the output type (named where the case is made).
"""
import importlib, importlib.util, math, os, sys, tempfile
import torch
import triton
import triton.language as tl
from triton_corpus import Case, gen, sentinel, SHORT

INTERP = os.environ.get('TRITON_INTERPRET') == '1'

# ------------------------------------------------------------------------------------------------
# Helpers


_PATCHED = {}


def _is_kernel(v):
    return type(v).__name__ in ('JITFunction', 'InterpretedFunction', 'Autotuner', 'Heuristics')


def patched(modname, fixes):
    """The module `modname`; under the interpreter, a copy of it whose source has each (old, new)
    of `fixes` replaced (each `old` must be found: a kernel that changed is seen), and whose kernels
    take the place of the module's own (so that the module's launchers, and the modules that
    imported the launchers, launch the copies). On the GPU the module itself, unchanged."""
    orig = importlib.import_module(modname)
    if not INTERP:
        return orig
    key = (modname, tuple(fixes))
    if key in _PATCHED:
        return _PATCHED[key]
    src = open(orig.__file__).read()
    for old, new in fixes:
        assert old in src, f'{modname}: "{old}" is not in the source any more'
        src = src.replace(old, new)
    # In a file of its own: the interpreter reads a kernel's source again from its file.
    d = tempfile.mkdtemp(prefix='vfvendor-')
    path = os.path.join(d, modname.replace('.', '_') + '.py')
    with open(path, 'w') as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(modname + '_interpreted', path)
    m = importlib.util.module_from_spec(spec)
    m.__package__ = orig.__package__
    spec.loader.exec_module(m)
    for k, v in list(vars(m).items()):
        if _is_kernel(v) and _is_kernel(getattr(orig, k, None)):
            setattr(orig, k, v)
    _PATCHED[key] = orig
    return orig


def pin(kernel, index=0, **want):
    """Leave each autotuner on the way to the JIT function of `kernel` one configuration: the first
    whose kwargs contain `want`, else number `index`. Returns the kernel."""
    k = kernel
    while k is not None and not isinstance(k, (triton.runtime.jit.JITFunction,)) and hasattr(k, 'fn'):
        if isinstance(k, triton.runtime.autotuner.Autotuner):
            if not hasattr(k, '_vf_all_configs'):
                k._vf_all_configs = list(k.configs)
            cfgs = k._vf_all_configs
            pick = [c for c in cfgs if all(c.kwargs.get(n) == v for n, v in want.items())] if want else []
            k.configs = [pick[0] if pick else cfgs[min(index, len(cfgs) - 1)]]
            k.cache = {}
        k = k.fn
    return kernel


def fill_like(t, v):
    return torch.full_like(t, v)


def tol_of(dtype, f16=2 ** -8, bf16=2 ** -6, f32=2 ** -14):
    """('tol', rtol, atol) for an output of `dtype` (the values used unless a case says otherwise)."""
    e = {torch.float16: f16, torch.bfloat16: bf16, torch.float32: f32}[dtype]
    return ('tol', e, e)


def cu_lens(lens):
    """Cumulative lengths with a leading 0, int32."""
    return torch.tensor([0] + list(itertools_accumulate(lens)), dtype=torch.int32)


def itertools_accumulate(xs):
    s, out = 0, []
    for x in xs:
        s += x
        out.append(s)
    return out


def paged_layout(seq_lens, block_size, g, extra_blocks=3):
    """A block table that scatters each sequence over shuffled pages: (block_table int32
    [num_seqs, max_blocks], number of pages)."""
    nblk = [triton.cdiv(s, block_size) for s in seq_lens]
    total = sum(nblk) + extra_blocks
    perm = torch.randperm(total, generator=g).to(torch.int32)
    width = max(max(nblk), 1)
    bt = torch.zeros(len(seq_lens), width, dtype=torch.int32)
    p = 0
    for i, n in enumerate(nblk):
        bt[i, :n] = perm[p:p + n]
        p += n
    return bt, total


def moe_align(topk_ids, num_experts, block_m, expert_map=None):
    """What vLLM's and SGLang's moe_align_block_size compute, on the CPU: the flat (token, k)
    indices sorted by expert and padded to a multiple of block_m with numel; the expert of each
    block (-1 for an expert of another rank when expert_map says so); the padded count."""
    flat = topk_ids.reshape(-1).to(torch.int64)
    n = flat.numel()
    max_padded = n + num_experts * (block_m - 1)
    sorted_ids = torch.full((max_padded,), n, dtype=torch.int32)
    expert_ids = torch.full((triton.cdiv(max_padded, block_m),), -1, dtype=torch.int32)
    pos = 0
    for e in range(num_experts):
        idx = (flat == e).nonzero().reshape(-1)
        cnt = idx.numel()
        if cnt == 0:
            continue
        padded = triton.cdiv(cnt, block_m) * block_m
        sorted_ids[pos:pos + cnt] = idx.to(torch.int32)
        eid = e if expert_map is None else int(expert_map[e])
        expert_ids[pos // block_m:(pos + padded) // block_m] = eid
        pos += padded
    return sorted_ids, expert_ids, torch.tensor([pos], dtype=torch.int32)


def topk_ids_of(g, M, E, topk):
    """Distinct experts per token, drawn at random: int32 [M, topk]."""
    return torch.stack([torch.randperm(E, generator=g)[:topk] for _ in range(M)]).to(torch.int32) if M else \
        torch.zeros(0, topk, dtype=torch.int32)


def dyadic(shape, g, choices=(0.25, 0.5, 1.0, 2.0)):
    """Powers of two chosen at random (scales and weights whose products are exact)."""
    c = torch.tensor(choices, dtype=torch.float64)
    return c[torch.randint(0, len(c), tuple(shape), generator=g)]


# ================================================================================================
# vLLM
# ================================================================================================


def vllm_fused_moe_cases():
    """fused_moe_kernel through vLLM's launcher invoke_fused_moe_triton_kernel (the alignment of
    moe_align_block_size is computed on the CPU: it is a CUDA extension in vLLM), and the
    GPTQ/AWQ kernel fused_moe_kernel_gptq_awq through invoke_fused_moe_wna16_triton_kernel. Small
    integers, dyadic routing weights and scales: exact."""
    from vllm.model_executor.layers.fused_moe import fused_moe as fm
    out = []
    dts = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}
    # (M tokens, E experts, top-k, N, K, dtype, BM, BN, BK, GROUP_M, warps, stages, routed weight, ep, bias)
    configs = [
        (1, 8, 2, 14336 // 4, 4096 // 4, torch.float16, 16, 64, 64, 1, 4, 2, False, False, False),
        (7, 8, 2, 1024, 512, torch.float16, 16, 32, 64, 1, 4, 3, True, False, False),
        (64, 8, 2, 2048, 1024, torch.float16, 64, 64, 32, 8, 4, 3, False, False, False),
        (33, 64, 6, 1408 // 2, 2048 // 2, torch.float16, 16, 64, 64, 1, 4, 2, True, False, False),
        (128, 64, 8, 768, 1024, torch.float16, 32, 64, 64, 8, 4, 4, False, False, True),
        (200, 16, 4, 520, 384, torch.float16, 64, 128, 32, 8, 8, 3, True, True, False),
        (16, 60, 4, 1408 // 4, 2048 // 2, torch.bfloat16, 16, 64, 64, 1, 4, 3, True, False, False),
        (96, 32, 8, 640, 512, torch.bfloat16, 64, 64, 64, 4, 4, 2, False, False, True),
        (40, 8, 2, 384, 288, torch.float32, 32, 32, 32, 8, 4, 2, True, False, False),
        (512, 8, 2, 1024, 256, torch.float16, 128, 128, 32, 8, 8, 3, True, False, False),
    ]
    for i, (M, E, topk, N, K, dt, bm, bn, bk, gm, nw, ns, mrw, ep, bias) in enumerate(configs):
        def make(g, M=M, E=E, topk=topk, N=N, K=K, dt=dt, bm=bm, ep=ep, bias=bias):
            ids = topk_ids_of(g, M, E, topk)
            emap = None
            if ep:
                # half the experts on another rank: their blocks write zeros (write_zeros_to_output)
                emap = torch.tensor([e if e % 2 == 0 else -1 for e in range(E)], dtype=torch.int32)
            s, eids, npad = moe_align(ids, E, bm, emap)
            t = {'a': gen(dt, (M, K), 'small', g), 'b': gen(dt, (E, N, K), 'small', g),
                 'w': dyadic((M, topk), g).to(torch.float32), 'sorted': s, 'eids': eids, 'npad': npad,
                 'c': sentinel(dt, (M, topk, N))}
            if bias:
                t['bias'] = gen(dt, (E, N), 'small', g)
            return t

        def launch(t, interpreted, topk=topk, dt=dt, bm=bm, bn=bn, bk=bk, gm=gm, nw=nw, ns=ns, mrw=mrw, bias=bias):
            cfg = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=gm, num_warps=nw, num_stages=ns)
            ct = dts[t['a'].dtype]
            fm.invoke_fused_moe_triton_kernel(t['a'], t['b'], t['c'], None, None, t['w'], t['sorted'], t['eids'],
                                              t['npad'], mrw, topk, cfg, ct, False, False, False, False, False,
                                              None, B_bias=t.get('bias'))
        name = f'vllm-fused_moe-{SHORT[dt]}-m{M}e{E}k{topk}-{N}x{K}-{bm}x{bn}x{bk}w{nw}s{ns}' + \
            ('-rw' if mrw else '') + ('-ep' if ep else '') + ('-bias' if bias else '')
        out.append(Case(name, fm.fused_moe_kernel, None, None, make, ['c'], launch=launch,
                        ref='interp32' if dt == torch.bfloat16 else 'interp'))

    # int8 weights, per-channel scale (use_int8_w8a16)
    for (M, E, topk, N, K, bm, bn, bk) in [(24, 8, 2, 512, 384, 16, 64, 64), (130, 16, 4, 384, 256, 64, 64, 32)]:
        def make(g, M=M, E=E, topk=topk, N=N, K=K, bm=bm):
            ids = topk_ids_of(g, M, E, topk)
            s, eids, npad = moe_align(ids, E, bm)
            return {'a': gen(torch.float16, (M, K), 'small', g), 'b': gen(torch.int8, (E, N, K), 'int', g),
                    'bs': dyadic((E, N), g).to(torch.float32), 'w': dyadic((M, topk), g).to(torch.float32),
                    'sorted': s, 'eids': eids, 'npad': npad, 'c': sentinel(torch.float16, (M, topk, N))}

        def launch(t, interpreted, topk=topk, bm=bm, bn=bn, bk=bk):
            cfg = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=1, num_warps=4, num_stages=3)
            fm.invoke_fused_moe_triton_kernel(t['a'], t['b'], t['c'], None, t['bs'], t['w'], t['sorted'], t['eids'],
                                              t['npad'], True, topk, cfg, tl.float16, False, False, True, False, False,
                                              None)
        out.append(Case(f'vllm-fused_moe-w8a16-m{M}e{E}k{topk}-{N}x{K}-{bm}x{bn}x{bk}', fm.fused_moe_kernel, None,
                        None, make, ['c'], launch=launch))

    # GPTQ/AWQ: int4 (packed two per byte along K) and int8 weights, group scales, zero points
    for (M, E, topk, N, K, group, bits, zp, bm) in [(1, 8, 2, 1024, 512, 128, 4, False, 16),
                                                    (19, 8, 2, 768, 1024, 128, 4, True, 16),
                                                    (70, 16, 4, 512, 256, 64, 4, False, 32),
                                                    (33, 8, 2, 384, 512, 128, 8, False, 16),
                                                    (48, 8, 2, 256, 256, 32, 8, True, 64)]:
        def make(g, M=M, E=E, topk=topk, N=N, K=K, group=group, bits=bits, zp=zp, bm=bm):
            ids = topk_ids_of(g, M, E, topk)
            s, eids, npad = moe_align(ids, E, bm)
            if bits == 4:
                q = torch.randint(0, 16, (E, N, K), generator=g, dtype=torch.int32)
                b = (q[:, :, 0::2] | (q[:, :, 1::2] << 4)).to(torch.uint8)
            else:
                b = torch.randint(0, 256, (E, N, K), generator=g, dtype=torch.int32).to(torch.uint8)
            t = {'a': gen(torch.float16, (M, K), 'small', g), 'b': b,
                 'bs': dyadic((E, N, K // group), g, (0.125, 0.25, 0.5)).to(torch.float16),
                 'w': dyadic((M, topk), g).to(torch.float32), 'sorted': s, 'eids': eids, 'npad': npad,
                 'c': sentinel(torch.float16, (M, topk, N))}
            if zp:
                if bits == 4:
                    z = torch.randint(0, 16, (E, N, K // group), generator=g, dtype=torch.int32)
                    t['zp'] = (z[:, 0::2, :] | (z[:, 1::2, :] << 4)).to(torch.uint8)
                else:
                    t['zp'] = torch.randint(100, 156, (E, N, K // group), generator=g, dtype=torch.int32).to(torch.uint8)
            return t

        def launch(t, interpreted, topk=topk, group=group, bits=bits, bm=bm):
            cfg = dict(BLOCK_SIZE_M=bm, GROUP_SIZE_M=1, SPLIT_K=1, num_warps=4, num_stages=2)
            fm.invoke_fused_moe_wna16_triton_kernel(t['a'], t['b'], t['c'], t['bs'], t.get('zp'), t['w'], t['sorted'],
                                                    t['eids'], t['npad'], True, topk, cfg, tl.float16, bits == 8,
                                                    bits == 4, [0, group])
        out.append(Case(f'vllm-fused_moe_wna16-int{bits}{"-zp" if zp else ""}-m{M}e{E}k{topk}-{N}x{K}-g{group}',
                        fm.fused_moe_kernel_gptq_awq, None, None, make, ['c'], launch=launch))

    # compute_identity_kernel (zero experts of LongCat): through zero_experts_compute_triton
    for (M, H, topk, E) in [(37, 2048, 4, 8), (1, 768, 2, 4), (256, 1024, 8, 16)]:
        def make(g, M=M, H=H, topk=topk, E=E):
            # about half the choices are zero experts (index E and above)
            return {'idx': torch.randint(0, 2 * E, (M, topk), generator=g, dtype=torch.int32),
                    'sc': dyadic((M, topk), g).to(torch.float32), 'h': gen(torch.float32, (M, H), 'int', g),
                    'o': sentinel(torch.float32, (M, H))}

        def launch(t, interpreted, E=E):
            t['o'].copy_(fm.zero_experts_compute_triton(t['idx'].clone(), t['sc'].clone(), E, 'identity', t['h']))
        out.append(Case(f'vllm-moe_identity-m{M}h{H}k{topk}', fm.compute_identity_kernel, None, None, make, ['o'],
                        launch=launch))
    return out


def vllm_unified_attention_cases():
    """kernel_unified_attention (the 2D launch, and the 3D launch with reduce_segments) through
    vLLM's launcher unified_attention, on a paged KV cache whose pages are shuffled. Random data:
    within 2^-8 (float16) or 2^-6 (bfloat16) of the interpreter, relative and absolute."""
    from vllm.v1.attention.ops import triton_unified_attention as ua
    out = []
    # (query lens, kv lens, Hq, Hkv, D, page, dtype, 3D, window, softcap, sinks, alibi)
    configs = [
        ([1] * 8, [300, 17, 1, 64, 1000, 33, 511, 128], 32, 8, 128, 16, torch.float16, False, -1, 0, False, False),
        ([1] * 16, [1024, 77, 2048, 5, 640, 100, 32, 64, 1, 900, 250, 333, 16, 48, 1500, 4], 32, 8, 128, 16,
         torch.float16, True, -1, 0, False, False),
        ([37, 1, 100, 5], [37, 300, 612, 5], 16, 2, 64, 32, torch.float16, False, -1, 0, False, False),
        ([64, 1, 1, 17], [128, 256, 33, 17], 28, 4, 128, 16, torch.bfloat16, False, -1, 0, False, False),
        ([1] * 6, [700, 64, 65, 1, 1500, 222], 28, 4, 128, 16, torch.bfloat16, True, -1, 0, False, False),
        ([20, 1, 9], [500, 400, 40], 8, 8, 96, 16, torch.float16, False, 127, 0, False, False),
        ([1] * 4, [513, 1024, 31, 260], 16, 4, 256, 16, torch.float16, True, 255, 50.0, False, False),
        ([12, 1, 1], [80, 1000, 65], 64, 8, 64, 16, torch.float16, False, -1, 0, True, False),
        ([1] * 5, [100, 3000, 7, 160, 48], 16, 16, 128, 32, torch.float16, False, -1, 0, False, True),
        ([300], [300], 32, 32, 64, 16, torch.float16, False, -1, 0, False, False),
    ]
    for i, (ql, kl, Hq, Hkv, D, page, dt, use3d, win, softcap, sinks, alibi) in enumerate(configs):
        def make(g, ql=ql, kl=kl, Hq=Hq, Hkv=Hkv, D=D, page=page, dt=dt, sinks=sinks, alibi=alibi):
            bt, nblocks = paged_layout(kl, page, g)
            T = sum(ql)
            t = {'q': gen(dt, (T, Hq, D), 'rnd', g), 'k': gen(dt, (nblocks, page, Hkv, D), 'rnd', g),
                 'v': gen(dt, (nblocks, page, Hkv, D), 'rnd', g), 'o': sentinel(dt, (T, Hq, D)),
                 'cu': cu_lens(ql), 'kl': torch.tensor(kl, dtype=torch.int32), 'bt': bt}
            if sinks:
                t['sinks'] = gen(torch.float32, (Hq,), 'rnd', g)
            if alibi:
                t['alibi'] = torch.tensor([2.0 ** (-8 * (h + 1) / Hq) for h in range(Hq)], dtype=torch.float32)
            return t

        def launch(t, interpreted, ql=ql, kl=kl, Hq=Hq, D=D, use3d=use3d, win=win, softcap=softcap):
            T = sum(ql)
            seg = {}
            if use3d:
                ns = 16
                dp = triton.next_power_of_2(D)
                dev = t['q'].device
                seg = dict(seq_threshold_3D=128, num_par_softmax_segments=ns,
                           softmax_segm_output=torch.zeros(T, Hq, ns, dp, dtype=torch.float32, device=dev),
                           softmax_segm_max=torch.zeros(T, Hq, ns, dtype=torch.float32, device=dev),
                           softmax_segm_expsum=torch.zeros(T, Hq, ns, dtype=torch.float32, device=dev))
            ua.unified_attention(t['q'], t['k'], t['v'], t['o'], t['cu'], max(ql), t['kl'], max(kl), 1.0 / math.sqrt(D),
                                 True, (win, 0) if win >= 0 else (-1, -1), t['bt'], softcap, None, None, None,
                                 alibi_slopes=t.get('alibi'), sinks=t.get('sinks'), **seg)
        name = (f'vllm-unified_attn-{"3d" if use3d else "2d"}-{SHORT[dt]}-s{len(ql)}q{sum(ql)}-h{Hq}kv{Hkv}d{D}p{page}'
                + (f'-w{win}' if win >= 0 else '') + ('-cap' if softcap else '') + ('-sink' if sinks else '')
                + ('-alibi' if alibi else ''))
        out.append(Case(name, ua.kernel_unified_attention, None, None, make, ['o'], launch=launch,
                        cmp=tol_of(dt), ref='interp32' if dt == torch.bfloat16 else 'interp'))
    return out


def no_rocm_query():
    """Under the interpreter, a stand-in for vllm.platforms.rocm, whose import asks the CUDA driver
    for the GPU's architecture (there is none): the launchers that import it only ask whether the
    GPU is an AMD RDNA one, which it is not on the GPU either."""
    if not INTERP or 'vllm.platforms.rocm' in sys.modules:
        return
    try:
        import vllm.platforms.rocm  # noqa: F401
    except Exception:
        import types
        m = types.ModuleType('vllm.platforms.rocm')
        m.on_rdna = lambda: False
        m.use_rocm_custom_paged_attention = lambda *a, **k: False
        sys.modules['vllm.platforms.rocm'] = m


_SHIMMED = []


def vllm_shims():
    """Under the interpreter (no GPU), answers to the questions vLLM's launchers ask the device, the
    ones a V100 gives on the GPU side, so that both sides choose the same launch: the device name
    (configuration files), 80 compute units, no pinned memory (it needs a driver), not an AMD GPU.
    Also the aliases of Triton functions in the flash-linear-attention modules (`exp = tl.exp`),
    which the interpreter does not replace by its own: they become JIT functions that call them."""
    if not INTERP or _SHIMMED:
        return
    _SHIMMED.append(1)
    no_rocm_query()
    from vllm.platforms import current_platform
    import vllm.utils.torch_utils as tu
    import vllm.utils.platform_utils as pu
    tu.PIN_MEMORY = False
    current_platform.get_device_name = lambda device_id=0: 'Tesla V100-SXM2-32GB'
    try:
        pu.get_device_name_as_file_name.cache_clear()
    except Exception:
        pass
    orig = pu.num_compute_units
    for m in list(sys.modules.values()):
        if getattr(m, 'num_compute_units', None) is orig:
            m.num_compute_units = lambda device_id=0: 80
    current_platform.num_compute_units = lambda device_id=0: 80
    fla_aliases('vllm.third_party.flash_linear_attention')


@triton.jit
def _jit_exp(x):
    return tl.exp(x)


@triton.jit
def _jit_exp2(x):
    return tl.exp2(x)


@triton.jit
def _jit_log(x):
    return tl.log(x)


@triton.jit
def _jit_log2(x):
    return tl.log2(x)


def fla_aliases(prefix):
    """Under the interpreter: in every loaded module under `prefix`, a name bound to tl.exp, tl.exp2,
    tl.log or tl.log2 is bound to a JIT function that calls it (the interpreter replaces the
    functions of `tl` while a kernel runs, not other names for them)."""
    if not INTERP:
        return
    import importlib as il
    try:
        il.import_module(prefix + '.ops.op')
    except Exception:
        pass
    table = {tl.exp: _jit_exp, tl.exp2: _jit_exp2, tl.log: _jit_log, tl.log2: _jit_log2}
    for name, m in list(sys.modules.items()):
        if not name.startswith(prefix) or m is None:
            continue
        for k in ('exp', 'exp2', 'log', 'log2'):
            v = getattr(m, k, None)
            if v is not None and v in table:
                setattr(m, k, table[v])


def paged_5d(g, dt, nblocks, Hkv, D, bs, kind='rnd'):
    """vLLM's paged cache of the prefix-prefill kernels: K [blocks, heads, D/x, block, x] (x
    elements make 16 bytes) and V [blocks, heads, D, block]."""
    x = 16 // torch.empty(0, dtype=dt).element_size()
    return gen(dt, (nblocks, Hkv, D // x, bs, x), kind, g), gen(dt, (nblocks, Hkv, D, bs), kind, g)


def vllm_attention_cases():
    """The other attention kernels of vLLM: prefix_prefill (_fwd_kernel, _fwd_kernel_alibi) through
    context_attention_fwd; chunked_prefill_paged_decode (kernel_paged_attention_2d and the prefix
    kernel); triton_decode_attention (_fwd_kernel_stage1, _fwd_grouped_kernel_stage1,
    _fwd_kernel_stage2) through decode_attention_fwd, with MHA, GQA and MLA (576/512) on pages of
    1, 16 and 64 tokens; triton_prefill_attention._fwd_kernel; merge_attn_states_kernel (with
    infinite LSEs); reshape_and_cache_kernel_flash. Random data within 2^-8 (float16) / 2^-6
    (bfloat16); the copy of reshape_and_cache bit for bit."""
    vllm_shims()
    from vllm.v1.attention.ops import prefix_prefill as pp
    from vllm.v1.attention.ops import chunked_prefill_paged_decode as cpd
    from vllm.v1.attention.ops import triton_decode_attention as tda
    from vllm.v1.attention.ops import triton_prefill_attention as tpa
    from vllm.v1.attention.ops import triton_merge_attn_states as tms
    from vllm.v1.attention.ops import triton_reshape_and_cache_flash as trc
    out = []

    # prefix prefill: query lens, context lens (the cached part before them), heads, D, block size
    for i, (ql, cl, Hq, Hkv, D, bs, dt, alibi, win, sinks) in enumerate([
            ([17, 64, 1, 130], [0, 100, 31, 513], 32, 8, 128, 16, torch.float16, False, 0, False),
            ([128, 33], [256, 1000], 16, 16, 64, 32, torch.float16, False, 0, False),
            ([40, 9, 200], [48, 0, 7], 8, 2, 128, 16, torch.bfloat16, False, 0, False),
            ([77, 5], [300, 64], 16, 4, 64, 16, torch.float16, True, 0, False),
            ([96, 50], [500, 3], 32, 8, 128, 16, torch.float16, False, 128, True)]):
        def make(g, ql=ql, cl=cl, Hq=Hq, Hkv=Hkv, D=D, bs=bs, dt=dt, alibi=alibi, sinks=sinks):
            sl = [a + b for a, b in zip(ql, cl)]
            bt, nb = paged_layout(sl, bs, g)
            kc, vc = paged_5d(g, dt, nb, Hkv, D, bs)
            T = sum(ql)
            t = {'q': gen(dt, (T, Hq, D), 'rnd', g), 'k': gen(dt, (T, Hkv, D), 'rnd', g),
                 'v': gen(dt, (T, Hkv, D), 'rnd', g), 'kc': kc, 'vc': vc, 'bt': bt, 'cu': cu_lens(ql),
                 'sl': torch.tensor(sl, dtype=torch.int32), 'o': sentinel(dt, (T, Hq, D)),
                 'ks': torch.ones(1, dtype=torch.float32), 'vs': torch.ones(1, dtype=torch.float32)}
            if alibi:
                t['alibi'] = torch.tensor([2.0 ** (-8 * (h + 1) / Hq) for h in range(Hq)], dtype=torch.float32)
            if sinks:
                t['sinks'] = gen(torch.float32, (Hq,), 'rnd', g)
            return t

        def launch(t, interpreted, ql=ql, cl=cl, win=win):
            pp.context_attention_fwd(t['q'], t['k'], t['v'], t['o'], 'auto', t['kc'], t['vc'], t['bt'], t['cu'],
                                     t['sl'], max(a + b for a, b in zip(ql, cl)), max(ql), t['ks'], t['vs'],
                                     alibi_slopes=t.get('alibi'), sliding_window=win, sinks=t.get('sinks'))
        out.append(Case(f'vllm-prefix_prefill-{SHORT[dt]}-s{len(ql)}q{sum(ql)}-h{Hq}kv{Hkv}d{D}b{bs}'
                        + ('-alibi' if alibi else '') + (f'-w{win}' if win else '') + ('-sink' if sinks else ''),
                        pp._fwd_kernel_alibi if alibi else pp._fwd_kernel, None, None, make, ['o'], launch=launch,
                        cmp=tol_of(dt), ref='interp32' if dt == torch.bfloat16 else 'interp'))

    # chunked prefill + paged decode: a mix of decodes (query 1) and prefills
    for (ql, cl, Hq, Hkv, D, bs, dt, win) in [([1, 1, 1, 30, 1, 64], [100, 5, 1023, 70, 0, 17], 32, 8, 128, 16,
                                                torch.float16, 0),
                                               ([1] * 12, [64, 3, 2000, 31, 32, 511, 200, 9, 1, 96, 128, 700], 16, 2,
                                                64, 32, torch.float16, 0),
                                               ([1, 1, 12], [300, 40, 100], 8, 8, 128, 16, torch.bfloat16, 64)]:
        def make(g, ql=ql, cl=cl, Hq=Hq, Hkv=Hkv, D=D, bs=bs, dt=dt):
            sl = [a + b for a, b in zip(ql, cl)]
            bt, nb = paged_layout(sl, bs, g)
            kc, vc = paged_5d(g, dt, nb, Hkv, D, bs)
            T = sum(ql)
            return {'q': gen(dt, (T, Hq, D), 'rnd', g), 'k': gen(dt, (T, Hkv, D), 'rnd', g),
                    'v': gen(dt, (T, Hkv, D), 'rnd', g), 'kc': kc, 'vc': vc, 'bt': bt, 'cu': cu_lens(ql),
                    'sl': torch.tensor(sl, dtype=torch.int32), 'o': sentinel(dt, (T, Hq, D)),
                    'ks': torch.ones(1, dtype=torch.float32), 'vs': torch.ones(1, dtype=torch.float32)}

        def launch(t, interpreted, ql=ql, cl=cl, win=win):
            no_rocm_query()
            cpd.chunked_prefill_paged_decode(t['q'], t['k'], t['v'], t['o'], 'auto', t['kc'], t['vc'], t['bt'], t['cu'],
                                             t['sl'], max(a + b for a, b in zip(ql, cl)), max(ql), t['ks'], t['vs'],
                                             sliding_window=win)
        out.append(Case(f'vllm-chunked_paged_decode-{SHORT[dt]}-s{len(ql)}q{sum(ql)}-h{Hq}kv{Hkv}d{D}b{bs}'
                        + (f'-w{win}' if win else ''), cpd.kernel_paged_attention_2d, None, None, make, ['o'],
                        launch=launch, cmp=tol_of(dt), ref='interp32' if dt == torch.bfloat16 else 'interp'))

    # decode attention (the Triton MLA backend's kernels): batch, Hq, Hkv, Dk, Dv, page, splits, cap, window
    for i, (kl, Hq, Hkv, Dk, Dv, page, splits, cap, win, dt) in enumerate([
            ([100, 1, 2047, 64, 513, 3000, 17, 256], 32, 32, 128, 128, 1, 8, 0.0, 0, torch.float16),
            ([100, 1, 2047, 64, 513, 3000, 17, 256], 32, 8, 128, 128, 16, 8, 0.0, 0, torch.float16),
            ([33] * 3 + [1000] * 3, 16, 1, 576, 512, 64, 4, 0.0, 0, torch.float16),
            ([700, 5, 64, 1024], 128, 1, 576, 512, 1, 16, 0.0, 0, torch.bfloat16),
            ([250, 70, 1], 28, 4, 128, 128, 16, 4, 30.0, 0, torch.float16),
            ([900, 129, 64, 2], 16, 4, 64, 64, 16, 8, 0.0, 100, torch.float16)]):
        mla = Hkv == 1 and Dk == 576

        def make(g, kl=kl, Hq=Hq, Hkv=Hkv, Dk=Dk, Dv=Dv, page=page, splits=splits, dt=dt, mla=mla):
            bt, nb = paged_layout(kl, page, g)
            B = len(kl)
            t = {'q': gen(dt, (B, Hq, Dk), 'rnd', g), 'kb': gen(dt, (nb, page, Hkv, Dk), 'rnd', g),
                 'bt': bt, 'sl': torch.tensor(kl, dtype=torch.int32), 'o': sentinel(dt, (B, Hq, Dv)),
                 'lse': sentinel(torch.float32, (B, Hq)),
                 'logits': torch.zeros(B, Hq, splits, Dv + 1, dtype=torch.float32)}
            if not mla:
                t['vb'] = gen(dt, (nb, page, Hkv, Dv), 'rnd', g)
            return t

        def launch(t, interpreted, splits=splits, page=page, cap=cap, win=win, Dk=Dk, Dv=Dv, mla=mla):
            vb = t['kb'][..., :Dv] if mla else t['vb']
            tda.decode_attention_fwd(t['q'], t['kb'], vb, t['o'], t['lse'], t['bt'], t['sl'], t['logits'], splits,
                                     1.0 / math.sqrt(Dk), page_size=page, logit_cap=cap, is_mla=mla,
                                     sliding_window=win or None)
        kind = 'mla' if mla else ('mha' if Hq == Hkv else 'gqa')
        out.append(Case(f'vllm-decode_attn-{kind}-{SHORT[dt]}-b{len(kl)}-h{Hq}kv{Hkv}d{Dk}-p{page}s{splits}'
                        + (f'-cap{int(cap)}' if cap else '') + (f'-w{win}' if win else ''),
                        tda._fwd_kernel_stage1 if Hq == Hkv else tda._fwd_grouped_kernel_stage1, None, None, make,
                        ['o', 'lse'], launch=launch, cmp=tol_of(dt, f32=2 ** -8) if dt != torch.float32 else tol_of(dt),
                        ref='interp32' if dt == torch.bfloat16 else 'interp'))

    # prefill attention (vision encoders, prefill of the Triton backends)
    for (sl, H, Hkv, D, dt, causal, win) in [([100, 1, 333, 64], 16, 16, 64, torch.float16, True, 0),
                                             ([1024, 77], 12, 12, 80, torch.float16, False, 0),
                                             ([500, 37], 32, 8, 128, torch.bfloat16, True, 0),
                                             ([256, 300], 16, 4, 128, torch.float16, True, 64)]:
        def make(g, sl=sl, H=H, Hkv=Hkv, D=D, dt=dt):
            T = sum(sl)
            return {'q': gen(dt, (T, H, D), 'rnd', g), 'k': gen(dt, (T, Hkv, D), 'rnd', g),
                    'v': gen(dt, (T, Hkv, D), 'rnd', g), 'o': sentinel(dt, (T, H, D)),
                    'st': cu_lens(sl)[:-1].contiguous(), 'sl': torch.tensor(sl, dtype=torch.int32)}

        def launch(t, interpreted, sl=sl, causal=causal, win=win):
            tpa.context_attention_fwd(t['q'], t['k'], t['v'], t['o'], t['st'], t['sl'], max(sl), is_causal=causal,
                                      sliding_window_q=win or None, sliding_window_k=win or None)
        out.append(Case(f'vllm-prefill_attn-{SHORT[dt]}-s{len(sl)}t{sum(sl)}-h{H}kv{Hkv}d{D}-c{int(causal)}'
                        + (f'-w{win}' if win else ''), tpa._fwd_kernel, None, None, make, ['o'], launch=launch,
                        cmp=tol_of(dt), ref='interp32' if dt == torch.bfloat16 else 'interp'))

    # merge_attn_states: prefix and suffix partial results with their LSE (some infinite)
    for (T, H, D, dt, with_lse, part) in [(500, 32, 128, torch.float16, True, None), (7, 8, 80, torch.bfloat16, False, None),
                                          (300, 16, 256, torch.float32, True, 200), (2048, 4, 64, torch.float16, True, None)]:
        def make(g, T=T, H=H, D=D, dt=dt):
            pl = gen(torch.float32, (H, T), 'rnd', g) * 4
            sl = gen(torch.float32, (H, T), 'rnd', g) * 4
            pick = torch.randint(0, 16, (H, T), generator=g)
            pl[pick == 0] = float('inf')
            pl[pick == 1] = float('-inf')
            sl[pick == 2] = float('-inf')
            return {'po': gen(dt, (T, H, D), 'rnd', g), 'so': gen(dt, (T, H, D), 'rnd', g), 'pl': pl, 'sl': sl,
                    'o': sentinel(dt, (T, H, D)), 'ol': sentinel(torch.float32, (H, T))}

        def launch(t, interpreted, with_lse=with_lse, part=part):
            tms.merge_attn_states(t['o'], t['po'], t['pl'], t['so'], t['sl'], t['ol'] if with_lse else None,
                                  prefill_tokens_with_context=part)
        out.append(Case(f'vllm-merge_attn_states-{SHORT[dt]}-t{T}h{H}d{D}' + ('-lse' if with_lse else '')
                        + (f'-ctx{part}' if part else ''), tms.merge_attn_states_kernel, None, None, make,
                        ['o', 'ol'] if with_lse else ['o'], launch=launch,
                        cmp=tol_of(dt, f32=2 ** -18) if dt == torch.float32 else tol_of(dt),
                        ref='interp32' if dt == torch.bfloat16 else 'interp'))

    # reshape_and_cache_flash: new keys and values into the paged cache, some slots -1 (padding)
    for (T, H, D, bs, nb, dt) in [(1, 8, 128, 16, 4, torch.float16), (300, 8, 128, 16, 40, torch.bfloat16),
                                  (77, 32, 64, 32, 10, torch.float16), (2048, 2, 256, 16, 200, torch.float32)]:
        def make(g, T=T, H=H, D=D, bs=bs, nb=nb, dt=dt):
            slots = torch.randperm(nb * bs, generator=g)[:T].to(torch.int64)
            slots[torch.randint(0, 9, (T,), generator=g) == 0] = -1
            return {'k': gen(dt, (T, H, D), 'rnd', g), 'v': gen(dt, (T, H, D), 'rnd', g), 'slots': slots,
                    'kc': sentinel(dt, (nb, bs, H, D)), 'vc': sentinel(dt, (nb, bs, H, D)),
                    'ks': torch.ones(1, dtype=torch.float32), 'vs': torch.ones(1, dtype=torch.float32)}

        def launch(t, interpreted, H=H, D=D, bs=bs):
            # the launcher's arguments, without its query of the device (capability below 9: tile 512)
            n = H * D
            kc = t['kc']
            grid = lambda meta: (t['slots'].shape[0], triton.cdiv(n, meta['TILE_SIZE']))
            trc.reshape_and_cache_kernel_flash[grid](
                key_ptr=t['k'], value_ptr=t['v'], key_cache_ptr=kc, value_cache_ptr=t['vc'],
                slot_mapping_ptr=t['slots'], k_scale=t['ks'], v_scale=t['vs'], key_stride=t['k'].stride(0),
                value_stride=t['v'].stride(0), block_stride=kc.stride(0), head_stride=kc.stride(2), dim_stride_k=0,
                dim_stride_v=0, page_stride=kc.stride(1), num_heads=H, head_size=D, block_size=bs, x=1,
                USE_HEAD_MAJOR_LAYOUT=False, INTERLEAVED_V_PACK_FACTOR=0, FP8_KV_CACHE=False,
                TILE_SIZE=min(512, triton.next_power_of_2(n)), num_warps=16, num_stages=10)
        out.append(Case(f'vllm-reshape_and_cache_flash-{SHORT[dt]}-t{T}h{H}d{D}b{bs}', trc.reshape_and_cache_kernel_flash,
                        None, None, make, ['kc', 'vc'], launch=launch,
                        note='launched with the arguments of triton_reshape_and_cache_flash, which asks the device'))
    return out


def vllm_sampling_cases():
    """The sampler of vLLM's GPU model runner (v1/worker/gpu/sample): penalties and their bincount
    (atomics on integers), min-p, temperature, Gumbel sampling, greedy argmax, logprobs and ranks,
    logit bias, bad words; top-k/top-p (v1/sample/ops/topk_topp_triton: the one-program-per-row
    kernel, and the split pipeline of small batches); the rejection sampler of speculative decoding.
    Vocabularies of 32000 to 151936."""
    vllm_shims()
    from vllm.v1.worker.gpu.sample import penalties as pen, min_p as mp, gumbel as gu, greedy_argmax as ga
    from vllm.v1.worker.gpu.sample import logprob as lp, logit_bias as lb, bad_words as bw
    from vllm.v1.sample.ops import topk_topp_triton as tt
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as rs
    out = []

    # penalties: logits of small integers, dyadic penalties: exact
    for (T, R, V, spec) in [(1, 4, 32000, 0), (64, 64, 152064, 0), (12, 3, 128256, 3), (5, 5, 32003, 0)]:
        def make(g, T=T, R=R, V=V, spec=spec):
            if spec:
                # requests of spec+1 tokens each (a sampled token and drafts): local positions 0..spec
                nreq = T // (spec + 1)
                idx = torch.arange(nreq, dtype=torch.int32).repeat_interleave(spec + 1)
                pos = torch.arange(spec + 1, dtype=torch.int32).repeat(nreq)
            else:
                idx = torch.randperm(R, generator=g)[:T].to(torch.int32) if T <= R else \
                    torch.randint(0, R, (T,), generator=g, dtype=torch.int32)
                pos = torch.zeros(T, dtype=torch.int32)
            counts = torch.zeros(R, V, dtype=torch.int32)
            hot = torch.randint(0, V, (R, 200), generator=g)
            counts.scatter_add_(1, hot, torch.ones_like(hot, dtype=torch.int32))
            words = triton.cdiv(V, 32)
            mask = torch.randint(-2 ** 31, 2 ** 31 - 1, (R, words), generator=g, dtype=torch.int64).to(torch.int32)
            mask &= torch.randint(-2 ** 31, 2 ** 31 - 1, (R, words), generator=g, dtype=torch.int64).to(torch.int32)
            rep = torch.tensor([1.0, 2.0, 0.5, 1.0, 4.0] * (R // 5 + 1))[:R]
            freq = torch.tensor([0.0, 0.5, 0.0, 0.25, 1.0] * (R // 5 + 1))[:R]
            pres = torch.tensor([0.0, 1.0, 2.0, 0.0, 0.5] * (R // 5 + 1))[:R]
            return {'logits': gen(torch.float32, (T, V), 'int', g), 'idx': idx, 'tok': torch.randint(
                0, V, (T,), generator=g, dtype=torch.int32), 'pos': pos, 'rep': rep, 'freq': freq, 'pres': pres,
                'mask': mask, 'counts': counts}

        def launch(t, interpreted):
            pen.apply_penalties(t['logits'], t['idx'], t['tok'], t['pos'], t['rep'], t['freq'], t['pres'], t['mask'],
                                t['counts'])
        out.append(Case(f'vllm-penalties-t{T}r{R}v{V}' + (f'-spec{spec}' if spec else ''), pen._penalties_kernel, None,
                        None, make, ['logits'], launch=launch))

    for (R, L, V) in [(8, 3000, 32000), (3, 70000, 151936), (64, 1100, 128256)]:
        def make(g, R=R, L=L, V=V):
            plen = torch.randint(1, L // 2, (R,), generator=g, dtype=torch.int32)
            flen = plen + torch.randint(0, L // 2, (R,), generator=g, dtype=torch.int32)
            return {'idx': torch.randperm(R, generator=g).to(torch.int32),
                    'tokens': torch.randint(0, V, (R, L), generator=g, dtype=torch.int32), 'plen': plen, 'flen': flen,
                    'mask': sentinel(torch.int32, (R, triton.cdiv(V, 32))), 'counts': sentinel(torch.int32, (R, V))}

        def launch(t, interpreted, L=L):
            pen.bincount(t['idx'], t['tokens'], t['plen'], t['flen'], t['mask'], t['counts'], L)
        out.append(Case(f'vllm-bincount-r{R}l{L}v{V}', pen._bincount_kernel, None, None, make, ['mask', 'counts'],
                        launch=launch, note='atomic or/add of integers: exact whatever their order'))

    # min-p and temperature (logits at random: the comparisons and the quotients by powers of two are exact)
    for (T, V) in [(1, 32000), (48, 151936), (7, 50257)]:
        def make(g, T=T, V=V):
            return {'logits': gen(torch.float32, (T, V), 'rnd', g) * 3, 'idx': torch.arange(T, dtype=torch.int32),
                    'minp': torch.tensor([0.0, 0.05, 0.1, 0.5, 0.01, 1.0, 0.2] * (T // 7 + 1))[:T].float(),
                    'temp': torch.tensor([1.0, 0.5, 0.0, 2.0, 0.25, 4.0, 0.125] * (T // 7 + 1))[:T].float()}

        def launch(t, interpreted):
            mp.apply_min_p(t['logits'], t['idx'], t['minp'])
            gu.apply_temperature(t['logits'], t['idx'], t['temp'])
        out.append(Case(f'vllm-min_p-temperature-t{T}v{V}', mp._min_p_kernel, None, None, make, ['logits'],
                        launch=launch))

    # Gumbel sampling (murmur3 hashes, logs): the sampled token
    for (T, V, temp_kind, fp64) in [(1, 32000, 'mix', False), (32, 151936, 'mix', False), (16, 128256, 'greedy', False),
                                    (9, 32000, 'mix', True)]:
        def make(g, T=T, V=V, temp_kind=temp_kind):
            temp = torch.tensor([0.7, 1.0, 0.0, 1.3] * (T // 4 + 1))[:T] if temp_kind == 'mix' else torch.zeros(T)
            return {'logits': gen(torch.float32, (T, V), 'rnd', g) * 2, 'idx': torch.arange(T, dtype=torch.int32),
                    'temp': temp.float(), 'seed': torch.randint(0, 2 ** 40, (T,), generator=g, dtype=torch.int64),
                    'pos': torch.randint(0, 4000, (T,), generator=g, dtype=torch.int64),
                    'sampled': sentinel(torch.int64, (T,))}

        def launch(t, interpreted, fp64=fp64):
            t['sampled'].copy_(gu.gumbel_sample(t['logits'], t['idx'], t['temp'], t['seed'], t['pos'], True, False,
                                                use_fp64=fp64))
        out.append(Case(f'vllm-gumbel_sample-t{T}v{V}-{temp_kind}' + ('-f64' if fp64 else ''), gu._gumbel_sample_kernel,
                        None, None, make, ['sampled'], launch=launch))

    # greedy argmax (launched as greedy_argmax launches it on a CUDA tensor), with NaN and ties
    for (T, V, dt) in [(1, 32000, torch.float32), (40, 151936, torch.bfloat16), (5, 4096, torch.float32)]:
        def make(g, T=T, V=V, dt=dt):
            x = gen(dt, (T, V), 'int', g)
            x[torch.randint(0, 50, (T, V), generator=g) == 0] = float('nan')
            return {'x': x, 'out': sentinel(torch.int64, (T,))}

        def launch(t, interpreted, T=T, V=V):
            x = t['x']
            splits = triton.cdiv(V, 4096)
            pv = torch.empty((T, splits), dtype=torch.float32, device=x.device)
            pi = torch.empty((T, splits), dtype=torch.int32, device=x.device)
            ga._argmax_partial[(T, splits)](x, pv, pi, V, x.stride(0), splits, 4096, num_warps=4)
            ga._argmax_finish[(T,)](pv, pi, t['out'], splits, triton.next_power_of_2(splits), num_warps=1)
        out.append(Case(f'vllm-greedy_argmax-t{T}v{V}-{SHORT[dt]}', ga._argmax_partial, None, None, make, ['out'],
                        launch=launch, ref='interp32' if dt == torch.bfloat16 else 'interp'))

    # logprobs of chosen tokens and the rank of the sampled token
    for (T, V, nlp) in [(1, 32000, 1), (32, 151936, 21), (8, 128256, 300)]:
        def make(g, T=T, V=V, nlp=nlp):
            return {'logits': gen(torch.float32, (T, V), 'rnd', g) * 4,
                    'ids': torch.randint(0, V, (T, nlp), generator=g, dtype=torch.int64),
                    'lp': sentinel(torch.float32, (T, nlp)), 'ranks': sentinel(torch.int64, (T,))}

        def launch(t, interpreted, T=T, V=V, nlp=nlp):
            t['lp'].copy_(lp.compute_token_logprobs(t['logits'], t['ids']))
            lp._ranks_kernel[(T,)](t['ranks'], t['logits'], t['logits'].stride(0), t['ids'][:, 0].contiguous(), V,
                                   BLOCK_SIZE=8192)
        out.append(Case(f'vllm-logprobs-ranks-t{T}v{V}n{nlp}', lp._topk_log_softmax_kernel, None, None, make,
                        ['lp', 'ranks'], launch=launch, cmp=('tol', 2 ** -18, 2 ** -16),
                        note='log-softmax: within 2^-16 absolute (a sum of exponentials in another order)'))

    # logit bias, allowed tokens, min tokens with stop tokens
    for (T, R, V) in [(6, 6, 32000), (40, 64, 151936)]:
        def make(g, T=T, R=R, V=V):
            nb = 37
            return {'logits': gen(torch.float32, (T, V), 'int', g), 'idx': torch.randperm(R, generator=g)[:T].to(torch.int32),
                    'pos': torch.randint(0, 20, (T,), generator=g, dtype=torch.int64),
                    'nallowed': torch.tensor([0, 5, 0, 37, 1, 0] * (R // 6 + 1))[:R].to(torch.int32),
                    'allowed': torch.randint(0, V, (R, nb), generator=g, dtype=torch.int32),
                    'nbias': torch.tensor([3, 0, 37, 1, 0, 2] * (R // 6 + 1))[:R].to(torch.int32),
                    'bias_ids': torch.randint(0, V, (R, nb), generator=g, dtype=torch.int32),
                    'bias': gen(torch.float32, (R, nb), 'int', g),
                    'minlen': torch.randint(0, 30, (R,), generator=g, dtype=torch.int32),
                    'nstop': torch.tensor([2, 0, 1, 5, 0, 3] * (R // 6 + 1))[:R].to(torch.int32),
                    'restore': torch.zeros(R, dtype=torch.bool),
                    'stop': torch.randint(0, V, (R, nb), generator=g, dtype=torch.int32)}

        def launch(t, interpreted):
            lb.apply_logit_bias(t['logits'], t['idx'], t['pos'], t['nallowed'], t['allowed'], t['nbias'], t['bias_ids'],
                                t['bias'], t['minlen'], t['nstop'], t['restore'], t['stop'])
        out.append(Case(f'vllm-logit_bias-t{T}r{R}v{V}', lb._bias_kernel, None, None, make, ['logits'], launch=launch))

    # top-k / top-p: the monolithic kernel (the interpreter needs one cast removed, as in
    # tests/gpu/vllm-topk-topp-test.py) and the split pipeline of batches up to 64 rows
    for i, (B, V, k, p) in enumerate([(4, 32000, 50, None), (16, 151936, 40, 0.9), (1, 128256, None, 0.95),
                                      (64, 32000, 1, 0.5), (3, 32000, 31999, None), (80, 50257, 200, 0.8),
                                      (2, 1000, 10, 0.3)]):
        def make(g, B=B, V=V, k=k, p=p):
            t = {'logits': gen(torch.float32, (B, V), 'rnd', g) * 2}
            if k is not None:
                t['k'] = torch.tensor([k, min(k * 2, V - 1), max(k // 2, 1), V] * (B // 4 + 1))[:B].to(torch.int32)
            if p is not None:
                t['p'] = torch.tensor([p, 0.99, 1.0, 0.3] * (B // 4 + 1))[:B].float()
            return t

        def launch(t, interpreted, B=B, V=V):
            m = patched('vllm.v1.sample.ops.topk_topp_triton', [('row_id.to(tl.int64)', 'row_id')])
            logits = t['logits']
            dev = logits.device
            nprog = min(80, B)
            buf = torch.zeros((nprog, V), dtype=torch.float32, device=dev)
            pct = torch.tensor(tt._PERCENTILE_TO_STD_TABLE, dtype=torch.float32, device=dev)
            cdf = torch.tensor(tt._NORMAL_CDF_TO_SIGMA_TABLE, dtype=torch.float32, device=dev)
            k, p = t.get('k'), t.get('p')
            split = p is not None and B <= 64
            m._topk_topp_kernel[(nprog,)](logits, logits.stride(0), buf, pct, cdf, k if k is not None else logits,
                                          p if p is not None else logits, B, VOCAB_SIZE=V, MASK_VALUE=float('-inf'),
                                          BLOCK_SIZE=8192, BLOCK_SIZE_TRUNC=4096, TOPK_ENABLED=k is not None,
                                          TOPP_ENABLED=p is not None, SPLIT_COVERS_PONLY=split, num_warps=8)
            if split:
                m._apply_topp_split(logits, k, p, float('-inf'), 80)
        out.append(Case(f'vllm-topk_topp-b{B}v{V}-k{k}-p{p}', tt._topk_topp_kernel, None, None, make, ['logits'],
                        launch=launch,
                        note='interpreter: row_id.to(tl.int64) -> row_id (row_id is a range variable)'))

    # rejection sampler (speculative decoding): greedy and random requests, draft tokens without draft logits
    for (R, steps, V) in [(4, 3, 32000), (32, 5, 151936), (1, 1, 128256)]:
        def make(g, R=R, steps=steps, V=V):
            n = R * (steps + 1)
            temp = torch.tensor([0.0, 0.8, 1.0, 0.0, 1.5] * (R // 5 + 1))[:R].float()
            logits = gen(torch.float32, (n, V), 'rnd', g) * 3
            draft = torch.randint(0, V, (n,), generator=g, dtype=torch.int64)
            # make many drafts the target's argmax, so that some are accepted
            am = logits.argmax(-1)
            take = torch.randint(0, 3, (n,), generator=g) > 0
            # the draft of a position is checked against the target's logits of the position before it
            prev = torch.roll(am, 1)
            first = torch.arange(n) % (steps + 1) == 0
            draft = torch.where(take & ~first, prev, draft)
            return {'logits': logits, 'draft': draft, 'cu': torch.arange(0, n + 1, steps + 1, dtype=torch.int32),
                    'pos': torch.randint(10, 4000, (n,), generator=g, dtype=torch.int64),
                    'idx': torch.arange(R, dtype=torch.int32),
                    'eidx': torch.arange(R, dtype=torch.int32).repeat_interleave(steps + 1),
                    'elp': torch.arange(steps + 1, dtype=torch.int32).repeat(R), 'temp': temp,
                    'seed': torch.randint(0, 2 ** 40, (R,), generator=g, dtype=torch.int64),
                    'sampled': sentinel(torch.int64, (R, steps + 1)), 'num': sentinel(torch.int32, (R,))}

        def launch(t, interpreted, steps=steps):
            s, n = rs.rejection_sample(t['logits'], None, t['draft'], t['cu'], t['pos'], t['idx'], t['eidx'], t['elp'],
                                       t['temp'], t['seed'], steps)
            # what lies beyond the sampled count of a request is not written: not compared
            keep = torch.arange(s.shape[1], device=s.device)[None, :] < n[:, None].to(torch.int64)
            t['sampled'].copy_(torch.where(keep, s.to(torch.int64), -1))
            t['num'].copy_(n)
        out.append(Case(f'vllm-rejection_sample-r{R}s{steps}v{V}', rs._rejection_kernel, None, None, make,
                        ['sampled', 'num'], launch=launch))
    return out


def lora_meta(g, M, L, none_frac=4):
    """vLLM's LoRA kernel metadata (LoRAKernelMeta.prepare_tensors) for M tokens over L adapters,
    some tokens without one (-1)."""
    mapping = torch.randint(0, L, (M,), generator=g, dtype=torch.int32)
    mapping[torch.randint(0, none_frac, (M,), generator=g) == 0] = -1
    _, srt = torch.sort(mapping, stable=True)
    ids, cnt = torch.unique(mapping, sorted=True, return_counts=True)
    n = ids.numel()
    lora_ids = torch.full((L + 1,), -1, dtype=torch.int32)
    lora_ids[:n] = ids
    ntok = torch.zeros(L + 1, dtype=torch.int32)
    ntok[:n] = cnt
    start = torch.zeros(L + 2, dtype=torch.int32)
    start[1:1 + n] = torch.cumsum(cnt, 0).to(torch.int32)
    return {'map': mapping, 'sorted': srt.to(torch.int32), 'ntok': ntok, 'start': start, 'ids': lora_ids,
            'nact': int(n)}


def pin_module(mod, index=0):
    """pin() every autotuned kernel of a module."""
    for v in list(vars(mod).values()):
        if isinstance(v, triton.runtime.autotuner.Autotuner) or (
                hasattr(v, 'fn') and isinstance(getattr(v, 'fn', None), triton.runtime.autotuner.Autotuner)):
            pin(v, index)


def lora_ref(slices, scale):
    """LoRA shrink then expand (add to the input) with PyTorch on the CPU: the shrink in float32,
    its result rounded to the weights' type before the expand, as the expand kernel casts it."""
    def ref(t):
        x, y = t['x'].float(), t['y'].float().clone()
        M = x.shape[0]
        mid = torch.zeros(slices, M, t['a0'].shape[1])
        col = 0
        for s in range(slices):
            a, b = t[f'a{s}'].float(), t[f'b{s}'].float()
            for m in range(M):
                lid = int(t['map'][m])
                if lid < 0:
                    continue
                mid[s, m] = (a[lid] @ x[m]) * scale
                y[m, col:col + b.shape[1]] += b[lid] @ mid[s, m].to(t['x'].dtype).float()
            col += b.shape[1]
        return {'mid': mid, 'y': y.to(t['y'].dtype)}
    return ref


def vllm_lora_cases():
    """The punica kernels of vLLM's LoRA: _lora_shrink_kernel and _lora_expand_kernel through
    _lora_shrink / _lora_expand, with 1 to 3 slices (qkv), ranks 8 to 64, tokens of several
    adapters and tokens without one. Small integers, scale 0.5: exact."""
    vllm_shims()
    from vllm.lora.ops.triton_ops import lora_shrink_op as so, lora_expand_op as eo
    out = []
    for i, (M, L, hidden, rank, slices, dt) in enumerate([(1, 4, 4096, 16, 1, torch.float16),
                                                          (64, 8, 4096, 8, 3, torch.float16),
                                                          (300, 4, 2048, 64, 2, torch.bfloat16),
                                                          (37, 16, 5120, 32, 1, torch.float16),
                                                          (512, 2, 1024, 16, 3, torch.bfloat16)]):
        def make(g, M=M, L=L, hidden=hidden, rank=rank, slices=slices, dt=dt):
            t = lora_meta(g, M, L)
            t['x'] = gen(dt, (M, hidden), 'small', g)
            for s in range(slices):
                t[f'a{s}'] = gen(dt, (L, rank, hidden), 'small', g)
                t[f'b{s}'] = gen(dt, (L, hidden // (s + 1), rank), 'small', g)
            t['mid'] = sentinel(torch.float32, (slices, M, rank))
            t['y'] = gen(dt, (M, sum(hidden // (s + 1) for s in range(slices))), 'small', g)
            return t

        def launch(t, interpreted, slices=slices):
            flag = torch.tensor([t['nact'] == 1 and int(t['ids'][0]) == -1])
            nact = torch.tensor([t['nact']])
            so._lora_shrink(t['x'], [t[f'a{s}'] for s in range(slices)], t['mid'], t['map'], t['sorted'], t['ntok'],
                            t['start'], t['ids'], flag, nact, 0.5)
            eo._lora_expand(t['mid'], [t[f'b{s}'] for s in range(slices)], t['y'], t['map'], t['sorted'], t['ntok'],
                            t['start'], t['ids'], flag, nact, offset_start=0, add_inputs=True)
        bf = dt == torch.bfloat16
        out.append(Case(f'vllm-lora-shrink_expand-{SHORT[dt]}-m{M}l{L}h{hidden}r{rank}x{slices}', so._lora_shrink_kernel,
                        None, None, make, ['mid', 'y'], launch=launch, ref=lora_ref(slices, 0.5) if bf else 'interp',
                        cmp=('tol', 2 ** -7, 2 ** -7) if bf else 'exact',
                        note='bfloat16: PyTorch on the CPU (the launchers refuse the float32 copies of interp32), '
                             'within 2^-7: a rounding to bfloat16 before or after adding the input' if bf else ''))
    return out


def mamba2_chunks(lens, chunk):
    """The chunk metadata of vLLM's Mamba2 prefill (no cached prefix): cu_chunk_seqlens,
    last_chunk_indices, seq_idx."""
    cu, last, sidx = [0], [], []
    pos = 0
    for b, n in enumerate(lens):
        end = pos + n
        while pos < end:
            pos = min(pos + chunk, end)
            cu.append(pos)
            sidx.append(b)
        last.append(len(cu) - 2)
    return (torch.tensor(cu, dtype=torch.int32), torch.tensor(last, dtype=torch.int32),
            torch.tensor(sidx, dtype=torch.int32))


def vllm_mamba_cases():
    """Mamba/Mamba2 kernels of vLLM: the SSD prefill chain (_chunk_cumsum_fwd_kernel,
    _bmm_chunk_fwd_kernel, _chunk_state_fwd_kernel, _state_passing_fwd_kernel,
    _chunk_scan_fwd_kernel) through mamba_chunk_scan_combined_varlen; the decode step
    (_selective_scan_update_kernel) through selective_state_update; the short convolution
    (_causal_conv1d_fwd_kernel, _causal_conv1d_update_kernel); the gated RMSNorm of Mamba2
    (_layer_norm_fwd_1pass_kernel). Random data: within 2^-10 relative and absolute for float32
    outputs, 2^-7 for float16/bfloat16 (exponentials and sums in another order)."""
    vllm_shims()
    from vllm.model_executor.layers.mamba.ops import ssd_combined as sc, mamba_ssm as ms, causal_conv1d as cc
    from vllm.model_executor.layers.mamba.ops import layernorm_gated as lg
    from vllm.model_executor.layers.mamba.ops import ssd_bmm, ssd_chunk_scan, ssd_chunk_state, ssd_state_passing
    out = []
    for m in (ssd_bmm, ssd_chunk_scan, ssd_chunk_state, ssd_state_passing, ms, cc):
        pin_module(m)

    for i, (lens, H, P, G, N, chunk, dt, init) in enumerate([([300, 1, 77], 16, 64, 2, 128, 256, torch.float16, False),
                                                             ([1000], 8, 64, 1, 64, 128, torch.float32, True),
                                                             ([129, 256, 33, 5], 32, 32, 8, 64, 64, torch.bfloat16, True)]):
        def make(g, lens=lens, H=H, P=P, G=G, N=N, chunk=chunk, dt=dt, init=init):
            T = sum(lens)
            cu_chunk, last, sidx = mamba2_chunks(lens, chunk)
            t = {'x': gen(dt, (T, H, P), 'rnd', g), 'dt': (gen(torch.float32, (T, H), 'rnd', g) * 0.5 - 1).to(dt),
                 'A': -torch.rand(H, generator=g).float() * 2 - 0.1, 'B': gen(dt, (T, G, N), 'rnd', g),
                 'C': gen(dt, (T, G, N), 'rnd', g), 'D': gen(torch.float32, (H,), 'rnd', g),
                 'z': gen(dt, (T, H, P), 'rnd', g), 'dt_bias': gen(torch.float32, (H,), 'rnd', g) * 0.1,
                 'cu': cu_lens(lens), 'cu_chunk': cu_chunk, 'last': last, 'sidx': sidx,
                 'out': sentinel(dt, (T, H, P)), 'states': sentinel(torch.float32, (len(lens) + 2, H, P, N)),
                 'slots': torch.randperm(len(lens) + 2, generator=g)[:len(lens)].to(torch.int32)}
            if init:
                t['h0'] = gen(torch.float32, (len(lens), H, P, N), 'rnd', g) * 0.5
            return t

        def launch(t, interpreted, chunk=chunk):
            sc.mamba_chunk_scan_combined_varlen(t['x'], t['dt'], t['A'], t['B'], t['C'], chunk, t['cu'], t['cu_chunk'],
                                                t['last'], t['sidx'], t['out'], D=t['D'], z=t['z'],
                                                dt_bias=t['dt_bias'], initial_states=t.get('h0'), dt_softplus=True,
                                                state_dtype=torch.float32, final_states_out=t['states'],
                                                final_state_indices=t['slots'])
        # bfloat16: the kernels round their products' operands to bfloat16, the reference (the
        # interpreter on float32 copies) does not: within 2^-3 + 2^-4 |ref| (measured: 0.066 (1 + |ref|))
        ot = {torch.float16: ('tol', 2 ** -7, 2 ** -7), torch.bfloat16: ('tol', 2 ** -4, 2 ** -3)}.get(dt, ('tol', 2 ** -10, 2 ** -10))
        out.append(Case(f'vllm-mamba2_ssd-{SHORT[dt]}-s{len(lens)}t{sum(lens)}-h{H}p{P}g{G}n{N}c{chunk}'
                        + ('-h0' if init else ''), ssd_chunk_scan._chunk_scan_fwd_kernel, None, None, make,
                        ['out', 'states'], launch=launch, cmp=ot, ref='interp32' if dt == torch.bfloat16 else 'interp'))

    # the decode step of Mamba2 (heads) and Mamba1 (no heads), with padded slots and a cache much larger
    for (B, H, P, G, N, dt, mamba1) in [(8, 16, 64, 8, 128, torch.float16, False), (1, 32, 64, 1, 128, torch.bfloat16, False),
                                        (40, 0, 1024, 1, 16, torch.float32, True), (32, 8, 64, 2, 64, torch.float16, False)]:
        def make(g, B=B, H=H, P=P, G=G, N=N, dt=dt, mamba1=mamba1):
            slots = B * 2 + 3
            idx = torch.randperm(slots, generator=g)[:B].to(torch.int32)
            if B > 2:
                idx[1] = -1   # a padded entry (PAD_SLOT_ID): not processed
            if mamba1:
                return {'state': gen(torch.float32, (slots, P, N), 'rnd', g), 'x': gen(dt, (B, P), 'rnd', g),
                        'dt': gen(dt, (B, P), 'rnd', g), 'A': -torch.rand(P, N, generator=g) - 0.1,
                        'B': gen(dt, (B, N), 'rnd', g), 'C': gen(dt, (B, N), 'rnd', g), 'D': gen(torch.float32, (P,), 'rnd', g),
                        'z': gen(dt, (B, P), 'rnd', g), 'dt_bias': gen(torch.float32, (P,), 'rnd', g) * 0.1, 'idx': idx,
                        'out': sentinel(dt, (B, P))}
            # Mamba2: A, D and dt_bias broadcast over the head dimension (strides 0), as vLLM passes them
            return {'state': gen(dt, (slots, H, P, N), 'rnd', g), 'x': gen(dt, (B, H, P), 'rnd', g),
                    'dt': gen(dt, (B, H), 'rnd', g), 'A': -torch.rand(H, generator=g) - 0.1,
                    'B': gen(dt, (B, G, N), 'rnd', g), 'C': gen(dt, (B, G, N), 'rnd', g), 'D': gen(torch.float32, (H,), 'rnd', g),
                    'z': gen(dt, (B, H, P), 'rnd', g), 'dt_bias': gen(torch.float32, (H,), 'rnd', g) * 0.1, 'idx': idx,
                    'out': sentinel(dt, (B, H, P))}

        def launch(t, interpreted, H=H, P=P, N=N, mamba1=mamba1):
            if mamba1:
                A, D, dtb, dtt = t['A'], t['D'], t['dt_bias'], t['dt']
            else:
                A = t['A'][:, None, None].expand(-1, P, N)
                D = t['D'][:, None].expand(-1, P)
                dtb = t['dt_bias'][:, None].expand(-1, P)
                dtt = t['dt'][:, :, None].expand(-1, -1, P)
            ms.selective_state_update(t['state'], t['x'], dtt, A, t['B'], t['C'], D, dtb, z=t['z'], dt_softplus=True,
                                      state_batch_indices=t['idx'], null_block_id=-1, out=t['out'])
        ot = ('tol', 2 ** -7, 2 ** -7) if dt != torch.float32 else ('tol', 2 ** -12, 2 ** -12)
        out.append(Case(f'vllm-selective_state_update-{"m1" if mamba1 else "m2"}-{SHORT[dt]}-b{B}h{H}p{P}g{G}n{N}',
                        ms._selective_scan_update_kernel, None, None, make, ['out', 'state'], launch=launch, cmp=ot,
                        ref='interp32' if dt == torch.bfloat16 else 'interp'))

    # causal conv1d: the prefill (varlen, continuous batching, with initial states) and the decode update
    for (lens, D, W, dt) in [([100, 1, 37, 300], 2048, 4, torch.float16), ([5000], 1024, 4, torch.bfloat16),
                             ([3, 2, 1, 64, 65, 7], 4096, 4, torch.float32)]:
        def make(g, lens=lens, D=D, W=W, dt=dt):
            T = sum(lens)
            nslots = len(lens) + 4
            # the programs: (sequence, block of 8 tokens), as the launcher's grid function lists them
            mlist = [b for b, n in enumerate(lens) for _ in range(triton.cdiv(n, 8))]
            offs = [j for n in lens for j in range(triton.cdiv(n, 8))]
            return {'x': gen(dt, (T, D), 'int', g).t(),  # (dim, tokens), channel last, as vLLM passes it
                    'w': gen(dt, (D, W), 'small', g), 'b': gen(dt, (D,), 'small', g),
                    'state': gen(dt, (nslots, W - 1, D), 'small', g).transpose(1, 2),
                    'cu': cu_lens(lens), 'idx': torch.randperm(nslots, generator=g)[:len(lens)].to(torch.int32),
                    'init': torch.tensor([i % 2 == 0 for i in range(len(lens))]),
                    'bp': torch.tensor(mlist, dtype=torch.int32), 'tco': torch.tensor(offs, dtype=torch.int32),
                    'o': sentinel(dt, (T, D)).t()}

        def launch(t, interpreted, D=D, W=W):
            x, st, o = t['x'], t['state'], t['o']
            grid = (t['bp'].numel(), triton.cdiv(D, 256))
            cc._causal_conv1d_fwd_kernel[grid](
                x, t['w'], t['b'], st, t['idx'], t['init'], t['cu'], t['bp'], t['tco'], o, D, st.size(0),
                x.stride(0), x.stride(1), t['w'].stride(0), t['w'].stride(1), st.stride(0), st.stride(1), st.stride(2),
                t['idx'].stride(0), o.stride(0), o.stride(1), -1, -1, HAS_BIAS=True, KERNEL_WIDTH=W,
                SILU_ACTIVATION=False, HAS_NULL_BLOCK=True, NP2_STATELEN=triton.next_power_of_2(W - 1), BLOCK_M=8,
                BLOCK_N=256, launch_pdl=False, num_stages=2)
        out.append(Case(f'vllm-causal_conv1d_fn-{SHORT[dt]}-s{len(lens)}t{sum(lens)}d{D}w{W}', cc._causal_conv1d_fwd_kernel,
                        None, None, make, ['o', 'state'], launch=launch,
                        ref='interp32' if dt == torch.bfloat16 else 'interp',
                        note='launched with the arguments of causal_conv1d_fn (whose grid function fills the program '
                             'table in place, which the interpreter cannot); no activation, small integers: exact'))
    for (B, D, W, L, dt) in [(8, 4096, 4, 1, torch.float16), (64, 2048, 4, 1, torch.bfloat16), (5, 1024, 4, 4, torch.float32)]:
        def make(g, B=B, D=D, W=W, L=L, dt=dt):
            nslots = B + 7
            return {'x': gen(dt, (B, D, L), 'int', g) if L > 1 else gen(dt, (B, D), 'int', g),
                    'w': gen(dt, (D, W), 'small', g), 'b': gen(dt, (D,), 'small', g),
                    'state': gen(dt, (nslots, D, W - 1 + L - 1), 'small', g),
                    'idx': (torch.randperm(nslots - 1, generator=g)[:B] + 1).to(torch.int32)}

        def launch(t, interpreted):
            t['x'].copy_(cc.causal_conv1d_update(t['x'], t['state'], t['w'], t['b'], activation=None,
                                                 conv_state_indices=t['idx']))
        out.append(Case(f'vllm-causal_conv1d_update-{SHORT[dt]}-b{B}d{D}w{W}l{L}', cc._causal_conv1d_update_kernel,
                        None, None, make, ['x', 'state'], launch=launch, ref='interp32' if dt == torch.bfloat16 else 'interp'))

    # gated RMSNorm (Mamba2), groups
    for (T, D, G, dt, before) in [(1, 8192, 8, torch.float16, False), (300, 4096, 1, torch.bfloat16, True),
                                  (77, 2048, 4, torch.float32, False)]:
        def make(g, T=T, D=D, dt=dt):
            return {'x': gen(dt, (T, D), 'rnd', g), 'z': gen(dt, (T, D), 'rnd', g), 'w': gen(dt, (D,), 'rnd', g),
                    'y': sentinel(dt, (T, D))}

        def launch(t, interpreted, D=D, G=G, before=before):
            t['y'].copy_(lg.rms_norm_gated(t['x'], t['w'], None, z=t['z'], eps=1e-5, group_size=D // G,
                                           norm_before_gate=before))
        out.append(Case(f'vllm-rms_norm_gated-{SHORT[dt]}-t{T}d{D}g{G}' + ('-before' if before else ''),
                        lg._layer_norm_fwd_1pass_kernel, None, None, make, ['y'], launch=launch,
                        cmp=tol_of(dt, f16=2 ** -9, bf16=2 ** -7, f32=2 ** -18),
                        ref='interp32' if dt == torch.bfloat16 else 'interp'))
    return out


def vllm_fla_cases():
    """The gated delta rule of Qwen3-Next (vllm/third_party/flash_linear_attention): the chunked
    prefill (chunk_local_cumsum, chunk_scaled_dot_kkt, solve_tril, recompute_w_u, chunk_delta_h,
    chunk_o kernels) through chunk_gated_delta_rule_fwd, the recurrent decode
    (fused_recurrent_gated_delta_rule), the fused sigmoid gating update, l2norm and the gated
    RMSNorm (layernorm_guard). Random data: within 2^-7 for float16 outputs, 2^-10 for float32."""
    vllm_shims()
    from vllm.third_party.flash_linear_attention.ops import chunk as fc, fused_recurrent as fr, l2norm as l2
    from vllm.third_party.flash_linear_attention.ops import fused_sigmoid_gating as fsg, layernorm_guard as lgd
    from vllm.third_party.flash_linear_attention.ops import (chunk_delta_h, chunk_o, chunk_scaled_dot_kkt, cumsum,
                                                             solve_tril, wy_fast)
    from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices, prepare_chunk_offsets
    fla_aliases('vllm.third_party.flash_linear_attention')
    out = []
    fix_h = [('i_t.to(tl.int64)', 'i_t')]
    for m in (fc, fr, l2, fsg, lgd, chunk_delta_h, chunk_o, chunk_scaled_dot_kkt, cumsum, solve_tril, wy_fast):
        pin_module(m)

    for i, (lens, H, K, dt, init) in enumerate([([100, 1, 200], 8, 128, torch.float16, True),
                                                ([1000], 4, 128, torch.float16, False),
                                                ([64, 65, 63, 17], 16, 64, torch.float32, True)]):
        def make(g, lens=lens, H=H, K=K, dt=dt, init=init):
            T = sum(lens)
            q = torch.nn.functional.normalize(gen(torch.float32, (1, T, H, K), 'rnd', g), dim=-1)
            k = torch.nn.functional.normalize(gen(torch.float32, (1, T, H, K), 'rnd', g), dim=-1)
            t = {'q': q.to(dt), 'k': k.to(dt), 'v': gen(dt, (1, T, H, K), 'rnd', g),
                 'g': torch.nn.functional.logsigmoid(gen(torch.float32, (1, T, H), 'rnd', g) + 3),
                 'beta': torch.rand(1, T, H, generator=g).float().to(dt), 'cu': cu_lens(lens),
                 'o': sentinel(dt, (1, T, H, K)), 'ht': sentinel(torch.float32, (len(lens), H, K, K))}
            if init:
                t['h0'] = gen(torch.float32, (len(lens), H, K, K), 'rnd', g) * 0.1
            return t

        def launch(t, interpreted):
            if interpreted:
                pin_module(patched('vllm.third_party.flash_linear_attention.ops.chunk_delta_h', fix_h))
                fla_aliases('vllm.third_party.flash_linear_attention')
            cu = t['cu']
            ci = prepare_chunk_indices(cu, 64)
            co = prepare_chunk_offsets(cu, 64)
            _, o, _, ht, *_ = fc.chunk_gated_delta_rule_fwd(t['q'], t['k'], t['v'], t['g'], t['beta'],
                                                            t['q'].shape[-1] ** -0.5, t.get('h0'), True, cu_seqlens=cu,
                                                            chunk_indices=ci, chunk_offsets=co)
            t['o'].copy_(o)
            t['ht'].copy_(ht)
        ot = ('tol', 2 ** -7, 2 ** -7) if dt != torch.float32 else ('tol', 2 ** -10, 2 ** -10)
        out.append(Case(f'vllm-gdn_chunk-{SHORT[dt]}-s{len(lens)}t{sum(lens)}h{H}k{K}' + ('-h0' if init else ''),
                        chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64, None, None, make, ['o', 'ht'],
                        launch=launch, cmp=ot,
                        note='interpreter: i_t.to(tl.int64) -> i_t in chunk_delta_h (i_t is a range variable)'))

    # recurrent decode: one token per sequence, state slots with padding (-1), GVA (HV = 2 H)
    for (B, H, HV, K, dt, l2n) in [(8, 16, 32, 128, torch.float16, True), (1, 4, 8, 128, torch.float32, False),
                                   (64, 2, 4, 64, torch.float16, True)]:
        def make(g, B=B, H=H, HV=HV, K=K, dt=dt):
            slots = B + 5
            # slot 0 is vLLM's null block (NULL_BLOCK_ID): a sequence there is skipped
            idx = (torch.randperm(slots - 1, generator=g)[:B] + 1).to(torch.int32)
            return {'q': gen(dt, (1, B, H, K), 'rnd', g), 'k': gen(dt, (1, B, H, K), 'rnd', g),
                    'v': gen(dt, (1, B, HV, K), 'rnd', g),
                    'g': torch.nn.functional.logsigmoid(gen(torch.float32, (1, B, HV), 'rnd', g) + 3),
                    'beta': torch.rand(1, B, HV, generator=g).float().to(dt),
                    'state': gen(torch.float32, (slots, HV, K, K), 'rnd', g) * 0.1, 'idx': idx,
                    'cu': torch.arange(B + 1, dtype=torch.int32), 'o': sentinel(dt, (1, B, HV, K))}

        def launch(t, interpreted, l2n=l2n):
            o, _ = fr.fused_recurrent_gated_delta_rule(t['q'], t['k'], t['v'], t['g'], t['beta'],
                                                       initial_state=t['state'], inplace_final_state=True,
                                                       cu_seqlens=t['cu'], ssm_state_indices=t['idx'],
                                                       use_qk_l2norm_in_kernel=l2n)
            t['o'].copy_(o)
        ot = ('tol', 2 ** -7, 2 ** -7) if dt != torch.float32 else ('tol', 2 ** -10, 2 ** -10)
        out.append(Case(f'vllm-gdn_recurrent-{SHORT[dt]}-b{B}h{H}hv{HV}k{K}' + ('-l2' if l2n else ''),
                        fr.fused_recurrent_gated_delta_rule_fwd_kernel, None, None, make, ['o', 'state'], launch=launch,
                        cmp=ot))

    # fused sigmoid gating + delta rule update (decode of Qwen3-Next)
    for (B, H, HV, K, dt) in [(16, 16, 32, 128, torch.float16), (3, 4, 4, 64, torch.float32)]:
        def make(g, B=B, H=H, HV=HV, K=K, dt=dt):
            slots = B + 3
            return {'A_log': gen(torch.float32, (HV,), 'rnd', g) * 0.5, 'a': gen(dt, (B, HV), 'rnd', g),
                    'b': gen(dt, (B, HV), 'rnd', g), 'dt_bias': gen(torch.float32, (HV,), 'rnd', g) * 0.1,
                    'q': gen(dt, (1, B, H, K), 'rnd', g), 'k': gen(dt, (1, B, H, K), 'rnd', g),
                    'v': gen(dt, (1, B, HV, K), 'rnd', g), 'state': gen(torch.float32, (slots, HV, K, K), 'rnd', g) * 0.1,
                    'idx': (torch.randperm(slots - 1, generator=g)[:B] + 1).to(torch.int32),
                    'cu': torch.arange(B + 1, dtype=torch.int32), 'o': sentinel(dt, (1, B, HV, K))}

        def launch(t, interpreted):
            o, _ = fsg.fused_sigmoid_gating_delta_rule_update(
                t['A_log'], t['a'], t['b'], t['dt_bias'], t['q'], t['k'], t['v'], initial_state=t['state'],
                inplace_final_state=True, cu_seqlens=t['cu'], ssm_state_indices=t['idx'], use_qk_l2norm_in_kernel=True)
            t['o'].copy_(o)
        ot = ('tol', 2 ** -7, 2 ** -7) if dt != torch.float32 else ('tol', 2 ** -10, 2 ** -10)
        out.append(Case(f'vllm-gdn_sigmoid_gating-{SHORT[dt]}-b{B}h{H}hv{HV}k{K}',
                        fsg.fused_sigmoid_gating_delta_rule_update_kernel, None, None, make, ['o', 'state'],
                        launch=launch, cmp=ot))

    # l2norm and the gated RMSNorm of the GDN output
    for (T, D, dt) in [(1, 128, torch.float16), (1024, 128, torch.float16), (333, 256, torch.float32)]:
        def make(g, T=T, D=D, dt=dt):
            return {'x': gen(dt, (T, 16, D), 'rnd', g), 'z': gen(dt, (T, 16, D), 'rnd', g), 'w': gen(dt, (D,), 'rnd', g),
                    'y': sentinel(dt, (T, 16, D)), 'n': sentinel(dt, (T, 16, D))}

        def launch(t, interpreted, D=D):
            t['n'].copy_(l2.l2norm_fwd(t['x']))
            t['y'].copy_(lgd.rmsnorm_fn(t['x'].reshape(-1, D), t['w'], None, z=t['z'].reshape(-1, D), eps=1e-6,
                                        group_size=None, norm_before_gate=True).reshape(t['y'].shape))
        out.append(Case(f'vllm-l2norm-rmsnorm_gated-{SHORT[dt]}-t{T}d{D}', l2.l2norm_fwd_kernel, None, None, make,
                        ['n', 'y'], launch=launch, cmp=tol_of(dt, f16=2 ** -9, f32=2 ** -18)))
    return out


def awq_pack(q):
    """AWQ's packing of 4-bit values along the output dimension: 8 per int32, in the order
    0, 2, 4, 6, 1, 3, 5, 7 (q: [..., N] int32 in 0..15 -> [..., N // 8] int32)."""
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    q = q.reshape(*q.shape[:-1], -1, 8).to(torch.int64)
    packed = torch.zeros(q.shape[:-1], dtype=torch.int64)
    for i, o in enumerate(order):
        packed |= q[..., o] << (4 * i)
    packed = torch.where(packed >= 2 ** 31, packed - 2 ** 32, packed)
    return packed.to(torch.int32)


def quant_rows(g, M, N, group, dt):
    """Rows of integers in [-127, 127] times a power of two per group, each group holding +-127
    times it: the scale absmax/127 is that power of two and every quotient is exact."""
    v = torch.randint(-127, 128, (M, N // group, group), generator=g).double()
    v[:, :, 0] = torch.where(torch.rand(M, N // group, generator=g) < 0.5, 127.0, -127.0)
    v = v * dyadic((M, N // group, 1), g, (0.0625, 0.125, 0.25, 0.5, 1.0))
    return v.reshape(M, N).to(dt)


def int8_row_ref(t):
    """per_token_quant_int8 with PyTorch: scale absmax/127, quotient rounded (exact on quant_rows)."""
    x = t['x'].float()
    am = x.abs().amax(1, keepdim=True).clamp(min=1e-10)
    return {'q': torch.round(x * (127 / am)).to(torch.int8), 's': am / 127}


def vllm_quant_cases():
    """Quantization kernels of vLLM that run without CUDA extensions: AWQ (awq_dequantize_kernel,
    awq_gemm_kernel with split-K) through awq_dequantize_triton / awq_gemm_triton;
    scaled_mm_kernel through triton_scaled_mm (int8 and float16 operands, per-token and
    per-channel scales, bias); _per_token_quant_int8 and _per_token_group_quant_int8; and a
    float8 kernel (_per_token_group_quant_fp8), whose float8 e4m3 type Triton may refuse for
    sm_75 (then COMPILE). Small integers and powers of two: exact."""
    vllm_shims()
    from vllm.model_executor.layers.quantization import awq_triton as aw
    from vllm.model_executor.layers.quantization.compressed_tensors import triton_scaled_mm as tsm
    from vllm.model_executor.layers.quantization.utils import int8_utils as i8, fp8_utils as f8
    out = []
    for i, (M, K, N, G, split) in enumerate([(1, 2048, 2048, 128, 1), (37, 2048, 1024, 128, 4), (256, 1024, 768, 64, 2),
                                             (5, 512, 11008 // 4, 32, 8)]):
        def make(g, M=M, K=K, N=N, G=G):
            # the kernel accumulates in float16 (its output type), in an order the hardware chooses:
            # activations of -1, 0, 1 (one in ten nonzero) and unit scales keep every partial sum an
            # integer far below 2048, exact in float16 whatever the order
            x = torch.randint(-1, 2, (M, K), generator=g) * (torch.randint(0, 10, (M, K), generator=g) == 0)
            return {'x': x.to(torch.float16),
                    'qw': awq_pack(torch.randint(0, 16, (K, N), generator=g, dtype=torch.int32)),
                    'qz': awq_pack(torch.randint(0, 16, (K // G, N), generator=g, dtype=torch.int32)),
                    'sc': torch.ones(K // G, N, dtype=torch.float16),
                    'w': sentinel(torch.float16, (K, N)), 'y': sentinel(torch.float16, (M, N))}

        def launch(t, interpreted, split=split):
            t['w'].copy_(aw.awq_dequantize_triton(t['qw'], t['sc'], t['qz']))
            t['y'].copy_(aw.awq_gemm_triton(t['x'], t['qw'], t['sc'], t['qz'], split))
        out.append(Case(f'vllm-awq-m{M}k{K}n{N}g{G}s{split}', aw.awq_gemm_kernel, None, None, make, ['w', 'y'],
                        launch=launch))

    for i, (M, K, N, dt, per_tok, per_ch, bias, heur) in enumerate([
            (1, 4096, 4096, torch.int8, True, True, False, True), (100, 2048, 1536, torch.int8, True, False, True, True),
            (300, 1024, 512, torch.float16, False, True, True, True), (64, 512, 2048, torch.int8, False, False, False, False),
            (33, 768, 640, torch.float16, True, True, False, False)]):
        def make(g, M=M, K=K, N=N, dt=dt, per_tok=per_tok, per_ch=per_ch, bias=bias):
            t = {'a': gen(dt, (M, K), 'int' if dt == torch.int8 else 'small', g),
                 'b': gen(dt, (N, K), 'int' if dt == torch.int8 else 'small', g).t(),  # column major, as vLLM keeps it
                 'sa': dyadic((M if per_tok else 1, 1), g, (0.0078125, 0.015625, 0.03125)).float(),
                 'sb': dyadic((N if per_ch else 1, 1), g, (0.0078125, 0.015625, 0.03125)).float(),
                 'y': sentinel(torch.float16, (M, N))}
            if bias:
                t['bias'] = gen(torch.float16, (N,), 'small', g)
            return t

        def launch(t, interpreted, heur=heur):
            t['y'].copy_(tsm.triton_scaled_mm(t['a'], t['b'], t['sa'], t['sb'], torch.float16, bias=t.get('bias'),
                                              use_heuristic=heur, use_td=False))
        out.append(Case(f'vllm-scaled_mm-{SHORT[dt]}-m{M}k{K}n{N}' + ('-tok' if per_tok else '') + ('-ch' if per_ch else '')
                        + ('-bias' if bias else '') + ('' if heur else '-32x32x32'), tsm.scaled_mm_kernel, None, None,
                        make, ['y'], launch=launch))

    for (M, N, G, dt) in [(1, 4096, 128, torch.float16), (300, 7168, 128, torch.bfloat16), (64, 2048, 64, torch.float32),
                          (17, 5120, 5120, torch.float16)]:
        def make(g, M=M, N=N, G=G, dt=dt):
            return {'x': quant_rows(g, M, N, G, dt), 'q': sentinel(torch.int8, (M, N)),
                    's': sentinel(torch.float32, (M, N // G))}

        def launch(t, interpreted, M=M, N=N, G=G):
            x = t['x']
            if G == N:
                q, s = i8.per_token_quant_int8(x)
                t['q'].copy_(q)
                t['s'].copy_(s)
            else:
                i8._per_token_group_quant_int8[(M * N // G,)](x, t['q'], t['s'], G, G, 1e-10, int8_min=-128,
                                                              int8_max=127, BLOCK=triton.next_power_of_2(G),
                                                              num_warps=min(max(triton.next_power_of_2(G) // 256, 1), 8),
                                                              num_stages=1)
        out.append(Case(f'vllm-int8_quant-{SHORT[dt]}-m{M}n{N}g{G}', i8._per_token_group_quant_int8 if G != N else
                        i8._per_token_quant_int8, None, None, make, ['q', 's'], launch=launch,
                        ref=int8_row_ref if G == N else ('interp32' if dt == torch.bfloat16 else 'interp'),
                        note='the group kernel launched as the CPU path of per_token_group_quant_int8 launches it '
                             '(on CUDA vLLM calls its C++ kernel)' if G != N else
                        'PyTorch reference: the kernel calls tl.extra.cuda.libdevice.round, which the interpreter '
                        'does not provide'))

    # float8: Triton's fp8e4nv needs sm_89; whatever happens is recorded
    def make8(g):
        return {'x': quant_rows(g, 64, 2048, 128, torch.float16), 's': sentinel(torch.float32, (64, 16)),
                'q': torch.zeros(64, 2048, dtype=torch.float8_e4m3fn)}

    def launch8(t, interpreted):
        x = t['x']
        f8._per_token_group_quant_fp8[(64 * 16,)](x, t['q'], t['s'], 128, 2048, 2048, 1e-10, fp8_min=-448.0,
                                                  fp8_max=448.0, use_ue8m0=False, BLOCK=128, num_warps=1, num_stages=1)
    out.append(Case('vllm-fp8_group_quant-f16-m64n2048g128', f8._per_token_group_quant_fp8, None, None, make8, ['q', 's'],
                    launch=launch8, ref=None, note='float8 e4m3: no reference (the interpreter has no float8 '
                                                   'arithmetic); routes and runs only, or COMPILE'))
    return out


def vllm_runner_cases():
    """The input preparation of vLLM's GPU model runner (v1/worker/gpu/input_batch.py: prefill
    inputs, positions and sequence lengths, sampled and draft tokens, the expanded request map, the
    sampled and rejected counts, the post-update of the request state) and its block tables
    (block_table.py: _gather_block_tables_kernel, _compute_slot_mappings_kernel, with pointer
    tables). A step with requests in prefill (chunked) and in decode with draft tokens. Integers:
    exact."""
    vllm_shims()
    from vllm.v1.worker.gpu import input_batch as ib, block_table as btm
    out = []
    for i, (R, n, S, L, chunk) in enumerate([(8, 5, 0, 600, 64), (64, 40, 3, 3000, 512), (16, 1, 5, 256, 128)]):
        def make(g, R=R, n=n, S=S, L=L, chunk=chunk):
            idx = torch.randperm(R, generator=g)[:n].to(torch.int32)
            prefill = torch.randint(5, L // 2, (R,), generator=g, dtype=torch.int32)
            computed = torch.zeros(R, dtype=torch.int32)
            qlen, nlog = [], []
            for b in range(n):
                r = int(idx[b])
                if b % 2 == 0:   # in prefill, possibly chunked
                    c = int(torch.randint(0, int(prefill[r]), (1,), generator=g))
                    computed[r] = c
                    qlen.append(min(chunk, int(prefill[r]) - c))
                    nlog.append(1)
                else:            # in decode, with S draft tokens
                    computed[r] = int(prefill[r]) + int(torch.randint(0, L // 4, (1,), generator=g))
                    qlen.append(1 + S)
                    nlog.append(1 + S)
            T = sum(qlen)
            ns = torch.tensor([int(torch.randint(1, x + 1, (1,), generator=g)) for x in nlog], dtype=torch.int32)
            V = 4096
            return {'idx': idx, 'prefill': prefill, 'computed': computed, 'qsl': cu_lens(qlen), 'cul': cu_lens(nlog),
                    'tokens': torch.randint(0, V, (R, L), generator=g, dtype=torch.int32),
                    'last': torch.randint(0, V, (R,), generator=g, dtype=torch.int32),
                    'drafts': torch.randint(0, V, (R, max(S, 1)), generator=g, dtype=torch.int32),
                    'sampled': torch.randint(0, V, (n, S + 1), generator=g, dtype=torch.int32),
                    'ns': ns, 'total': prefill.clone(), 'nlog_total': sum(nlog), 'maxexp': max(nlog), 'S': S,
                    'ids': sentinel(torch.int32, (T,)), 'pos': sentinel(torch.int64, (T,)),
                    'seq': sentinel(torch.int32, (R,)), 'nxt': sentinel(torch.int32, (2, R)),
                    'li': sentinel(torch.int64, (sum(nlog),)), 'eidx': sentinel(torch.int32, (sum(nlog),)),
                    'elp': sentinel(torch.int32, (sum(nlog),)), 'nrej': sentinel(torch.int32, (n,)),
                    'counts': torch.zeros(R, V, dtype=torch.int32)}

        def launch(t, interpreted):
            ib.prepare_prefill_inputs(t['ids'], t['nxt'], t['idx'], t['qsl'], t['tokens'], t['prefill'], t['computed'])
            ib.prepare_pos_seq_lens(t['idx'], t['qsl'], t['computed'], t['pos'], t['seq'])
            S = t['S']
            t['li'].copy_(ib.combine_sampled_and_draft_tokens(t['ids'], t['idx'], t['last'], t['qsl'], t['seq'],
                                                              t['prefill'], t['drafts'][:, :S], t['cul'],
                                                              t['nlog_total']))
            e, lp = ib.expand_idx_mapping(t['idx'], t['nlog_total'], t['cul'], t['maxexp'])
            t['eidx'].copy_(e)
            t['elp'].copy_(lp)
            _, nr = ib.get_num_sampled_and_rejected(t['ns'], t['seq'], t['cul'], t['idx'], t['prefill'])
            t['nrej'].copy_(nr)
            ib.post_update(t['idx'], t['computed'], t['last'], t['counts'], t['sampled'], t['ns'], t['nrej'], t['qsl'],
                           t['tokens'], t['total'])
            ib.post_update_num_computed_tokens(t['idx'], t['computed'], t['qsl'])
        out.append(Case(f'vllm-input_batch-r{R}n{n}s{S}l{L}c{chunk}', ib._prepare_prefill_inputs_kernel, None, None, make,
                        ['ids', 'pos', 'seq', 'nxt', 'li', 'eidx', 'elp', 'ns', 'nrej', 'computed', 'last', 'tokens',
                         'total', 'counts'], launch=launch))

    # block tables of two KV cache groups (pages of 16 and of 64), gathered for a batch with padded
    # rows (zeroed), then the slot of every token
    for (R, n, pad, maxb, T) in [(16, 9, 3, 64, 300), (128, 100, 28, 512, 2000)]:
        def make(g, R=R, n=n, maxb=maxb, T=T):
            idx = torch.randperm(R, generator=g)[:n].to(torch.int32)
            nb = torch.stack([torch.randint(1, maxb + 1, (R,), generator=g),
                              torch.randint(1, maxb // 4 + 1, (R,), generator=g)]).to(torch.int32)
            q = torch.randint(1, max(2, 2 * T // n), (n,), generator=g)
            pos = []
            for b in range(n):
                span = 16 * maxb // 4   # positions within the pages of both groups
                start = int(torch.randint(0, max(1, span - int(q[b])), (1,), generator=g))
                pos.append(torch.arange(start, start + int(q[b])))
            return {'idx': idx, 'nb': nb, 'src0': torch.randint(0, 100000, (R, maxb), generator=g, dtype=torch.int32),
                    'src1': torch.randint(0, 100000, (R, maxb // 4), generator=g, dtype=torch.int32),
                    'dst0': sentinel(torch.int32, (R, maxb)), 'dst1': sentinel(torch.int32, (R, maxb // 4)),
                    'qsl': cu_lens(q.tolist()), 'pos': torch.cat(pos).to(torch.int64),
                    'slots': sentinel(torch.int64, (2, int(q.sum()) + 37)), 'computed': torch.zeros(R, dtype=torch.int32)}

        def launch(t, interpreted, n=n, pad=pad, maxb=maxb):
            dev = t['idx'].device
            src = torch.tensor([t['src0'].data_ptr(), t['src1'].data_ptr()], dtype=torch.uint64, device=dev)
            dst = torch.tensor([t['dst0'].data_ptr(), t['dst1'].data_ptr()], dtype=torch.uint64, device=dev)
            strides = torch.tensor([maxb, maxb // 4], dtype=torch.int32, device=dev)
            bsz = torch.tensor([16, 64], dtype=torch.int32, device=dev)
            reach = torch.zeros(2, dtype=torch.int32, device=dev)
            btm._gather_block_tables_kernel[(2, n + pad)](t['idx'], src, dst, strides, t['nb'], t['nb'].stride(0), n,
                                                          t['computed'], reach, bsz, MASK_SLIDING_WINDOW=False,
                                                          BLOCK_SIZE=1024)
            enabled = torch.ones(2, dtype=torch.bool, device=dev)
            sharded = torch.zeros(2, dtype=torch.bool, device=dev)
            btm._compute_slot_mappings_kernel[(2, n + 1)](t['slots'].shape[1], t['idx'], t['qsl'], t['pos'], src, strides,
                                                          bsz, enabled, sharded, t['slots'], t['slots'].stride(0), 0,
                                                          CP_SIZE=1, CP_INTERLEAVE=1, PAD_ID=-1, TRITON_BLOCK_SIZE=1024)
        out.append(Case(f'vllm-block_table-r{R}n{n}p{pad}b{maxb}t{T}', btm._gather_block_tables_kernel, None, None, make,
                        ['dst0', 'dst1', 'slots'], launch=launch,
                        note="launched with BlockTables' arguments (tables of pointers made from the tensors)"))
    return out


def vllm_batch_invariant_cases():
    """vLLM's batch-invariant operators (model_executor/determinism/batch_invariant.py): the
    persistent matmul (matmul_kernel_persistent), bmm_kernel, the softmax/log-softmax kernel and
    mean_kernel. Products of small integers: exact; softmax and means within 2^-16 relative."""
    vllm_shims()
    from vllm.model_executor.determinism import batch_invariant as bi
    out = []
    for i, (M, K, N, dt, bias) in enumerate([(1, 4096, 4096, torch.float16, False), (300, 1024, 2048, torch.bfloat16, True),
                                             (77, 512, 1000, torch.float32, False), (1024, 256, 512, torch.float16, True)]):
        def make(g, M=M, K=K, N=N, dt=dt, bias=bias):
            t = {'a': gen(dt, (M, K), 'small', g), 'b': gen(dt, (K, N), 'small', g), 'c': sentinel(dt, (M, N))}
            if bias:
                t['bias'] = gen(dt, (N,), 'small', g)
            return t

        def launch(t, interpreted, bias=bias):
            t['c'].copy_(bi.matmul_persistent(t['a'], t['b'], t.get('bias')))
        out.append(Case(f'vllm-bi_matmul-{SHORT[dt]}-m{M}k{K}n{N}' + ('-bias' if bias else ''), bi.matmul_kernel_persistent,
                        None, None, make, ['c'], launch=launch, ref='interp32' if dt == torch.bfloat16 else 'interp'))
    for (B, M, K, N, dt) in [(8, 128, 64, 128, torch.float16), (3, 77, 256, 33, torch.float32)]:
        def make(g, B=B, M=M, K=K, N=N, dt=dt):
            return {'a': gen(dt, (B, M, K), 'small', g), 'b': gen(dt, (B, K, N), 'small', g), 'c': sentinel(dt, (B, M, N))}

        def launch(t, interpreted):
            bi.bmm_batch_invariant(t['a'], t['b'], out=t['c'])
        out.append(Case(f'vllm-bi_bmm-{SHORT[dt]}-b{B}m{M}k{K}n{N}', bi.bmm_kernel, None, None, make, ['c'], launch=launch))
    for (M, N, dt) in [(64, 32000, torch.float32), (7, 151936, torch.float32), (300, 1000, torch.float16)]:
        def make(g, M=M, N=N, dt=dt):
            return {'x': gen(dt, (M, N), 'rnd', g) * 3, 'ls': sentinel(dt, (M, N)), 'sm': sentinel(dt, (M, N)),
                    'mean': sentinel(dt, (M,))}

        def launch(t, interpreted):
            t['ls'].copy_(bi.log_softmax(t['x'], -1))
            t['sm'].copy_(bi.softmax(t['x'], -1))
            t['mean'].copy_(bi.mean_dim(t['x'], 1))
        out.append(Case(f'vllm-bi_softmax_mean-{SHORT[dt]}-m{M}n{N}', bi._softmax_kernel, None, None, make,
                        ['ls', 'sm', 'mean'], launch=launch,
                        cmp=('tol', 2 ** -16, 2 ** -18) if dt == torch.float32 else tol_of(dt)))
    return out


def vllm_rope_cases():
    """Rotary embeddings of vLLM in Triton: triton_mrope (Qwen2-VL / Qwen3-VL, sections T/H/W,
    interleaved or not, NeoX or GPT-J pairs). Random data within 2^-9 (float16) / 2^-7 (bfloat16)."""
    vllm_shims()
    from vllm.model_executor.layers.rotary_embedding import mrope
    out = []
    for (T, Hq, Hkv, D, rd, sec, inter, neox, dt) in [(300, 28, 4, 128, 128, [16, 24, 24], False, True, torch.bfloat16),
                                                      (1, 16, 2, 128, 128, [24, 20, 20], True, True, torch.float16),
                                                      (77, 32, 8, 128, 64, [8, 12, 12], False, False, torch.float16)]:
        def make(g, T=T, Hq=Hq, Hkv=Hkv, D=D, rd=rd, dt=dt):
            ang = torch.rand(3, T, rd // 2, generator=g) * 6.28
            return {'q': gen(dt, (T, Hq * D), 'rnd', g), 'k': gen(dt, (T, Hkv * D), 'rnd', g),
                    'cos': ang.cos().to(dt), 'sin': ang.sin().to(dt)}

        def launch(t, interpreted, D=D, rd=rd, sec=sec, inter=inter, neox=neox):
            q, k = mrope.triton_mrope(t['q'], t['k'], t['cos'], t['sin'], sec, D, rd, inter, neox)
            t['q'].copy_(q)
            t['k'].copy_(k)
        out.append(Case(f'vllm-mrope-{SHORT[dt]}-t{T}h{Hq}kv{Hkv}d{D}r{rd}' + ('-inter' if inter else '')
                        + ('-neox' if neox else '-gptj'), getattr(mrope, '_triton_mrope_forward', None), None, None,
                        make, ['q', 'k'], launch=launch, cmp=tol_of(dt, f16=2 ** -9, bf16=2 ** -7),
                        ref='interp32' if dt == torch.bfloat16 else 'interp'))
    return out


VLLM_FAMILIES = [vllm_fused_moe_cases, vllm_unified_attention_cases, vllm_attention_cases, vllm_sampling_cases,
                 vllm_lora_cases, vllm_mamba_cases, vllm_fla_cases, vllm_quant_cases, vllm_runner_cases,
                 vllm_batch_invariant_cases, vllm_rope_cases]

# ================================================================================================
# SGLang
# ================================================================================================

_SGL_SHIMMED = []


def sgl_shims():
    """Under the interpreter, the answers a V100 gives to SGLang's launchers (80 multiprocessors),
    a no-op device guard for CPU tensors, and the flash-linear-attention aliases (see vllm_shims)."""
    if not INTERP or _SGL_SHIMMED:
        return
    _SGL_SHIMMED.append(1)
    import contextlib
    if not hasattr(torch.cpu, 'device'):
        # SGLang's device guard enters torch.<device type>.device(index); the CPU has none
        torch.cpu.device = lambda *a, **k: contextlib.nullcontext()
    import sglang.srt.utils.common as cm
    orig = cm.get_device_core_count
    for m in list(sys.modules.values()):
        if getattr(m, 'get_device_core_count', None) is orig:
            m.get_device_core_count = lambda device_id=0: 80
    fla_aliases('sglang.kernels.ops.attention.fla')


def sgl_attention_cases():
    """SGLang's Triton attention backend (sglang/kernels/ops/attention): decode_attention_fwd
    (_fwd_kernel_stage1 for MHA, _fwd_grouped_kernel_stage1 for GQA and MLA 576/512,
    _fwd_kernel_stage2) on a token pool (page 1) with shuffled slots; extend_attention_fwd
    (_fwd_kernel: the new tokens against a cached prefix, causal, with and without a sliding
    window); context_attention_fwd (prefill); merge_state_kernel (with infinite LSEs);
    create_flashinfer_kv_indices_triton. Random data within 2^-8 (float16) / 2^-6 (bfloat16);
    the indices bit for bit."""
    sgl_shims()
    from sglang.kernels.ops.attention import decode_attention as da, extend_attention as ea
    from sglang.kernels.ops.attention import prefill_attention as pa, merge_state as ms, kv_indices as ki
    sgl_shims()
    out = []

    def pool(g, lens, extra=7):
        """Token slots of each request in a pool larger than needed, shuffled: kv_indptr, kv_indices."""
        total = sum(lens)
        slots = torch.randperm(total + extra, generator=g)[:total].to(torch.int64) + 1
        return cu_lens(lens), slots.to(torch.int32), total + extra + 1

    for i, (kl, Hq, Hkv, Dk, Dv, splits, cap, dt) in enumerate([
            ([100, 1, 2047, 64, 513, 3000, 17, 256], 32, 32, 128, 128, 8, 0.0, torch.float16),
            ([100, 1, 2047, 64, 513, 3000, 17, 256], 32, 8, 128, 128, 8, 0.0, torch.float16),
            ([33, 1000, 5, 600], 16, 1, 576, 512, 4, 0.0, torch.bfloat16),
            ([250, 70, 1], 28, 4, 128, 128, 4, 30.0, torch.float16),
            ([900, 129, 64, 2] * 4, 8, 2, 64, 64, 16, 0.0, torch.bfloat16)]):
        mla = Dk == 576

        def make(g, kl=kl, Hq=Hq, Hkv=Hkv, Dk=Dk, Dv=Dv, splits=splits, dt=dt, mla=mla):
            indptr, idx, nslots = pool(g, kl)
            B = len(kl)
            t = {'q': gen(dt, (B, Hq, Dk), 'rnd', g), 'kb': gen(dt, (nslots, Hkv, Dk), 'rnd', g), 'indptr': indptr,
                 'idx': idx, 'o': sentinel(dt, (B, Hq, Dv)),
                 'logits': torch.zeros(B, Hq, splits, Dv, dtype=torch.float32),
                 'lse': torch.zeros(B, Hq, splits, dtype=torch.float32),
                 'nsplit': torch.tensor([min(splits, max(1, k // 64)) for k in kl], dtype=torch.int32)}
            if not mla:
                t['vb'] = gen(dt, (nslots, Hkv, Dv), 'rnd', g)
            return t

        def launch(t, interpreted, splits=splits, cap=cap, Dk=Dk, Dv=Dv, mla=mla):
            vb = t['kb'][..., :Dv] if mla else t['vb']
            da.decode_attention_fwd(t['q'], t['kb'], vb, t['o'], t['indptr'], t['idx'], t['logits'], t['lse'],
                                    t['nsplit'], splits, 1.0 / math.sqrt(Dk), 1.0, 1.0, logit_cap=cap, has_mla=mla)
        kind = 'mla' if mla else ('mha' if Hq == Hkv else 'gqa')
        out.append(Case(f'sgl-decode_attn-{kind}-{SHORT[dt]}-b{len(kl)}-h{Hq}kv{Hkv}d{Dk}-s{splits}'
                        + (f'-cap{int(cap)}' if cap else ''),
                        da._fwd_kernel_stage1 if Hq == Hkv else da._fwd_grouped_kernel_stage1, None, None, make, ['o'],
                        launch=launch, cmp=tol_of(dt), ref='interp32' if dt == torch.bfloat16 else 'interp'))

    # extend: new tokens of each request (q/k/v extend) against its cached prefix in the pool
    for i, (ql, pl, Hq, Hkv, D, dt, win) in enumerate([([17, 64, 1, 130], [0, 100, 31, 513], 32, 8, 128, torch.float16, -1),
                                                       ([128, 33], [256, 1000], 16, 16, 64, torch.float16, -1),
                                                       ([40, 9, 200], [48, 0, 7], 8, 2, 128, torch.bfloat16, -1),
                                                       ([96, 50], [500, 3], 32, 8, 128, torch.float16, 127)]):
        def make(g, ql=ql, pl=pl, Hq=Hq, Hkv=Hkv, D=D, dt=dt):
            indptr, idx, nslots = pool(g, pl)
            T = sum(ql)
            return {'q': gen(dt, (T, Hq, D), 'rnd', g), 'k': gen(dt, (T, Hkv, D), 'rnd', g),
                    'v': gen(dt, (T, Hkv, D), 'rnd', g), 'o': sentinel(dt, (T, Hq, D)),
                    'kb': gen(dt, (nslots, Hkv, D), 'rnd', g), 'vb': gen(dt, (nslots, Hkv, D), 'rnd', g),
                    'qo': cu_lens(ql), 'indptr': indptr, 'idx': idx}

        def launch(t, interpreted, ql=ql, win=win):
            ea.extend_attention_fwd(t['q'], t['k'], t['v'], t['o'], t['kb'], t['vb'], t['qo'], t['indptr'], t['idx'],
                                    None, True, None, max(ql), 1.0, 1.0, sliding_window_size=win)
        out.append(Case(f'sgl-extend_attn-{SHORT[dt]}-s{len(ql)}q{sum(ql)}p{sum(pl)}-h{Hq}kv{Hkv}d{D}'
                        + (f'-w{win}' if win >= 0 else ''), ea._fwd_kernel, None, None, make, ['o'], launch=launch,
                        cmp=tol_of(dt), ref='interp32' if dt == torch.bfloat16 else 'interp'))

    for (sl, H, Hkv, D, dt, causal) in [([100, 1, 333, 64], 16, 16, 64, torch.float16, True),
                                        ([1024, 77], 12, 12, 80, torch.float16, False),
                                        ([500, 37], 32, 8, 128, torch.bfloat16, True)]:
        def make(g, sl=sl, H=H, Hkv=Hkv, D=D, dt=dt):
            T = sum(sl)
            return {'q': gen(dt, (T, H, D), 'rnd', g), 'k': gen(dt, (T, Hkv, D), 'rnd', g),
                    'v': gen(dt, (T, Hkv, D), 'rnd', g), 'o': sentinel(dt, (T, H, D)),
                    'st': cu_lens(sl)[:-1].contiguous(), 'sl': torch.tensor(sl, dtype=torch.int32)}

        def launch(t, interpreted, sl=sl, causal=causal):
            pa.context_attention_fwd(t['q'], t['k'], t['v'], t['o'], t['st'], t['sl'], max(sl), is_causal=causal)
        out.append(Case(f'sgl-prefill_attn-{SHORT[dt]}-s{len(sl)}t{sum(sl)}-h{H}kv{Hkv}d{D}-c{int(causal)}', pa._fwd_kernel,
                        None, None, make, ['o'], launch=launch, cmp=tol_of(dt),
                        ref='interp32' if dt == torch.bfloat16 else 'interp'))

    for (T, H, D, dt) in [(500, 32, 128, torch.float16), (7, 8, 80, torch.bfloat16), (300, 16, 256, torch.float32)]:
        def make(g, T=T, H=H, D=D, dt=dt):
            pl = gen(torch.float32, (T, H), 'rnd', g) * 4
            sl = gen(torch.float32, (T, H), 'rnd', g) * 4
            pick = torch.randint(0, 16, (T, H), generator=g)
            pl[pick == 0] = float('inf')
            pl[pick == 1] = float('-inf')
            sl[pick == 2] = float('-inf')
            return {'po': gen(dt, (T, H, D), 'rnd', g), 'so': gen(dt, (T, H, D), 'rnd', g), 'pl': pl, 'sl': sl,
                    'o': sentinel(dt, (T, H, D)), 'ol': sentinel(torch.float32, (T, H))}

        def launch(t, interpreted):
            ms.merge_state_triton(t['po'], t['pl'], t['so'], t['sl'], t['o'], t['ol'])
        out.append(Case(f'sgl-merge_state-{SHORT[dt]}-t{T}h{H}d{D}', ms.merge_state_kernel, None, None, make, ['o', 'ol'],
                        launch=launch, cmp=tol_of(dt, f32=2 ** -18), ref='interp32' if dt == torch.bfloat16 else 'interp'))

    # flashinfer kv indices from the request-to-token table, with start offsets
    for (B, R, L, par) in [(8, 16, 4096, False), (64, 100, 2048, True), (1, 4, 70000, False)]:
        def make(g, B=B, R=R, L=L):
            lens = torch.randint(1, L, (B,), generator=g, dtype=torch.int32)
            start = (torch.randint(0, 64, (B,), generator=g, dtype=torch.int32)).clamp(max=L - lens)
            return {'table': torch.randint(0, 2 ** 30, (R, L), generator=g, dtype=torch.int32),
                    'rows': torch.randperm(R, generator=g)[:B].to(torch.int32), 'lens': lens,
                    'indptr': cu_lens(lens.tolist()), 'start': start,
                    'out': sentinel(torch.int32, (int(lens.sum()) + 5,))}

        def launch(t, interpreted, B=B, par=par):
            grid = (B, 8) if par else (B,)
            ki.create_flashinfer_kv_indices_triton[grid](t['table'], t['rows'], t['lens'], t['indptr'], t['start'],
                                                         t['out'], t['table'].stride(0), TOKEN_BLOCK_PARALLEL=par)
        out.append(Case(f'sgl-kv_indices-b{B}r{R}l{L}' + ('-par' if par else ''), ki.create_flashinfer_kv_indices_triton,
                        None, None, make, ['out'], launch=launch))
    return out


def sgl_moe_cases():
    """SGLang's fused MoE (sglang/kernels/ops/moe): fused_moe_kernel and fused_moe_kernel_gptq_awq
    through invoke_fused_moe_kernel (alignment computed on the CPU, as moe_align_block_size would);
    _moe_align_small_numel_kernel (fill then scatter across warps, with a barrier between);
    act_and_mul_kernel; _moe_sum_reduce_kernel. Small integers and powers of two: exact."""
    sgl_shims()
    from sglang.kernels.ops.moe import fused_moe_triton_kernels as fk, moe_align_small_numel as mas
    out = []
    dts = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}
    for i, (M, E, topk, N, K, dt, bm, bn, bk, gm, nw, ns, mrw, ep) in enumerate([
            (1, 8, 2, 14336 // 4, 4096 // 4, torch.float16, 16, 64, 64, 1, 4, 2, False, False),
            (64, 8, 2, 2048, 1024, torch.float16, 64, 64, 32, 8, 4, 3, False, False),
            (33, 64, 6, 1408 // 2, 2048 // 2, torch.float16, 16, 64, 64, 1, 4, 2, True, False),
            (200, 16, 4, 520, 384, torch.float16, 64, 128, 32, 8, 8, 3, True, True),
            (96, 32, 8, 640, 512, torch.bfloat16, 64, 64, 64, 4, 4, 2, False, False)]):
        def make(g, M=M, E=E, topk=topk, N=N, K=K, dt=dt, bm=bm, ep=ep):
            ids = topk_ids_of(g, M, E, topk)
            emap = torch.tensor([e if e % 2 == 0 else -1 for e in range(E)], dtype=torch.int32) if ep else None
            s, eids, npad = moe_align(ids, E, bm, emap)
            return {'a': gen(dt, (M, K), 'small', g), 'b': gen(dt, (E, N, K), 'small', g), 'ids': ids,
                    'w': dyadic((M, topk), g).to(torch.float32), 'sorted': s, 'eids': eids, 'npad': npad,
                    'c': sentinel(dt, (M, topk, N))}

        def launch(t, interpreted, topk=topk, bm=bm, bn=bn, bk=bk, gm=gm, nw=nw, ns=ns, mrw=mrw):
            cfg = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=gm, num_warps=nw, num_stages=ns)
            fk.invoke_fused_moe_kernel(t['a'], t['b'], None, t['c'], None, None, None, t['w'], t['ids'], t['sorted'],
                                       t['eids'], t['npad'], mrw, topk, cfg, dts[t['a'].dtype], False, False, False,
                                       False, False)
        out.append(Case(f'sgl-fused_moe-{SHORT[dt]}-m{M}e{E}k{topk}-{N}x{K}-{bm}x{bn}x{bk}w{nw}s{ns}'
                        + ('-rw' if mrw else '') + ('-ep' if ep else ''), fk.fused_moe_kernel, None, None, make, ['c'],
                        launch=launch, ref='interp32' if dt == torch.bfloat16 else 'interp'))

    for (M, E, topk, N, K, group, bits, zp, bm) in [(19, 8, 2, 768, 1024, 128, 4, True, 16),
                                                    (70, 16, 4, 512, 256, 64, 4, False, 32),
                                                    (33, 8, 2, 384, 512, 128, 8, False, 16)]:
        def make(g, M=M, E=E, topk=topk, N=N, K=K, group=group, bits=bits, zp=zp, bm=bm):
            ids = topk_ids_of(g, M, E, topk)
            s, eids, npad = moe_align(ids, E, bm)
            if bits == 4:
                q = torch.randint(0, 16, (E, N, K), generator=g, dtype=torch.int32)
                b = (q[:, :, 0::2] | (q[:, :, 1::2] << 4)).to(torch.uint8)
            else:
                b = torch.randint(0, 256, (E, N, K), generator=g, dtype=torch.int32).to(torch.uint8)
            t = {'a': gen(torch.float16, (M, K), 'small', g), 'b': b, 'ids': ids,
                 'bs': dyadic((E, N, K // group), g, (0.125, 0.25, 0.5)).to(torch.float16),
                 'w': dyadic((M, topk), g).to(torch.float32), 'sorted': s, 'eids': eids, 'npad': npad,
                 'c': sentinel(torch.float16, (M, topk, N))}
            if zp:
                z = torch.randint(0, 16, (E, N, K // group), generator=g, dtype=torch.int32)
                t['zp'] = (z[:, 0::2, :] | (z[:, 1::2, :] << 4)).to(torch.uint8)
            return t

        def launch(t, interpreted, topk=topk, group=group, bits=bits, bm=bm):
            cfg = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=1, num_warps=4, num_stages=2)
            fk.invoke_fused_moe_kernel(t['a'], t['b'], None, t['c'], None, t['bs'], t.get('zp'), t['w'], t['ids'],
                                       t['sorted'], t['eids'], t['npad'], True, topk, cfg, tl.float16, False, False,
                                       bits == 8, bits == 4, False, block_shape=[0, group])
        out.append(Case(f'sgl-fused_moe_wna16-int{bits}{"-zp" if zp else ""}-m{M}e{E}k{topk}-{N}x{K}-g{group}',
                        fk.fused_moe_kernel_gptq_awq, None, None, make, ['c'], launch=launch))

    # the single-launch alignment of tiny batches (numel <= 64), experts -1 filtered
    for (M, E, topk, bs) in [(1, 256, 8, 16), (8, 128, 8, 64), (3, 512, 6, 16), (16, 64, 4, 32)]:
        def make(g, M=M, E=E, topk=topk, bs=bs):
            ids = topk_ids_of(g, M, E, topk).reshape(-1)
            ids[torch.randint(0, 6, ids.shape, generator=g) == 0] = -1
            numel = ids.numel()
            maxpad = numel + (E + 1) * (bs - 1)
            return {'ids': ids, 'sorted': sentinel(torch.int32, (maxpad,)),
                    'eids': sentinel(torch.int32, (triton.cdiv(maxpad, bs),)), 'npad': sentinel(torch.int32, (1,))}

        def launch(t, interpreted, E=E, bs=bs):
            mas.moe_align_small_numel(t['ids'], E + 1, bs, t['sorted'], t['eids'], t['npad'])
        out.append(Case(f'sgl-moe_align_small-m{M}e{E}k{topk}b{bs}', mas._moe_align_small_numel_kernel, None, None, make,
                        ['sorted', 'eids', 'npad'], launch=launch))

    for (T, I, act, dt) in [(300, 1408, 'silu', torch.float16), (64, 2048, 'gelu', torch.bfloat16),
                            (7, 768, 'silu', torch.float32)]:
        def make(g, T=T, I=I, dt=dt):
            return {'x': gen(dt, (T, 2 * I), 'rnd', g), 'y': sentinel(dt, (T, I))}

        def launch(t, interpreted, act=act):
            fk.act_and_mul_triton(t['x'], t['y'], {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, activation=act)
        out.append(Case(f'sgl-act_and_mul-{act}-{SHORT[dt]}-t{T}i{I}', fk.act_and_mul_kernel, None, None, make, ['y'],
                        launch=launch, cmp=tol_of(dt, f16=2 ** -9, bf16=2 ** -7, f32=2 ** -18),
                        ref='interp32' if dt == torch.bfloat16 else 'interp'))
    for (T, topk, H, dt) in [(300, 8, 2048, torch.float16), (1, 2, 7168, torch.bfloat16), (64, 6, 4096, torch.float32)]:
        def make(g, T=T, topk=topk, H=H, dt=dt):
            return {'x': gen(dt, (T, topk, H), 'small', g), 'y': sentinel(dt, (T, H))}

        def launch(t, interpreted):
            fk.moe_sum_reduce_triton(t['x'], t['y'], 0.5)
        out.append(Case(f'sgl-moe_sum_reduce-{SHORT[dt]}-t{T}k{topk}h{H}', fk._moe_sum_reduce_kernel, None, None, make,
                        ['y'], launch=launch, ref='interp32' if dt == torch.bfloat16 else 'interp'))
    return out


def sgl_misc_cases():
    """SGLang's sampling kernels (temperature softmax in three kernels, top-p/top-k renormalisation,
    softcapped row argmax), KV-cache and memory kernels (reshape_and_cache_flash, the page copy of
    the allocator, the copy of KV slots over all layers through a table of pointers,
    memcpy_triton), RMSNorm (fused_rmsnorm_kernel, fused_dual_residual_rmsnorm_kernel with up to 32
    warps, _rmsnorm_fp32_kernel) and gelu_and_mul_kernel. Launchers that fall back to PyTorch on
    a CPU tensor are bypassed: the kernels are launched with their arguments (said per case)."""
    sgl_shims()
    from sglang.kernels.ops.sampling import temperature_softmax as ts, renorm_triton as rn, softcap_row_argmax as sra
    from sglang.kernels.ops.kvcache import cache_ops as co, copy_pages as cp, cache_move as cm
    from sglang.kernels.ops.memory import memcpy_triton as mc
    from sglang.kernels.ops.layernorm import fused_rmsnorm as fr, rmsnorm_fp32 as r32
    from sglang.kernels.ops.activation import gelu_and_mul_triton as gm
    out = []

    for (R, V) in [(1, 151936), (16, 32000), (4, 128256)]:
        def make(g, R=R, V=V):
            return {'x': gen(torch.float32, (R, V), 'rnd', g) * 3, 't': dyadic((R,), g, (0.5, 1.0, 2.0)).float(),
                    'p': sentinel(torch.float32, (R, V))}

        def launch(t, interpreted, R=R, V=V):
            x, tt, p = t['x'], t['t'], t['p']
            block, ns = ts._split_geometry(R, V)
            dev = x.device
            pm = torch.empty((R, ns), dtype=torch.float32, device=dev)
            psum = torch.empty((R, ns), dtype=torch.float32, device=dev)
            rm = torch.empty((R,), dtype=torch.float32, device=dev)
            rs_ = torch.empty((R,), dtype=torch.float32, device=dev)
            ts._partial_max_sumexp_kernel[(R, ns)](x, tt, pm, psum, x.stride(0), V, ns, BLOCK=block, num_warps=8)
            ts._combine_kernel[(R,)](pm, psum, rm, rs_, ns, BLOCK=triton.next_power_of_2(ns), num_warps=1)
            ts._normalize_kernel[(R, ns)](x, tt, rm, rs_, p, x.stride(0), p.stride(0), V, BLOCK=block, num_warps=8)
        out.append(Case(f'sgl-temperature_softmax-r{R}v{V}', ts._partial_max_sumexp_kernel, None, None, make, ['p'],
                        launch=launch, cmp=('tol', 2 ** -16, 2 ** -30),
                        note='the three launches of temperature_softmax (which falls back to PyTorch on the CPU)'))

    for (R, V) in [(8, 32000), (2, 151936)]:
        def make(g, R=R, V=V):
            pr = torch.softmax(gen(torch.float32, (R, V), 'rnd', g) * 3, -1)
            piv = torch.sort(pr, -1, descending=True).values[torch.arange(R), torch.randint(1, 400, (R,), generator=g)]
            return {'p': pr, 'piv': piv.contiguous(), 'o': sentinel(torch.float32, (R, V))}

        def launch(t, interpreted):
            t['o'].copy_(rn._renorm_from_pivots(t['p'], t['piv']))
        out.append(Case(f'sgl-renorm-r{R}v{V}', rn._mask_and_partial_sum_kernel, None, None, make, ['o'], launch=launch,
                        cmp=('tol', 2 ** -18, 2 ** -30)))

    for (R, V, cap) in [(8, 32000, 30.0), (3, 262144, 30.0), (64, 4096, 50.0)]:
        def make(g, R=R, V=V):
            return {'x': gen(torch.float32, (R, V + 64), 'rnd', g) * 5, 'o': sentinel(torch.float32, (R, 2))}

        def launch(t, interpreted, V=V, cap=cap):
            t['o'].copy_(sra.softcap_row_argmax(t['x'], cap, V, 1000))

        def ref(t, V=V, cap=cap):
            v = cap * torch.tanh(t['x'][:, :V].double() / cap)
            m, i = v.max(1)
            return {'o': torch.stack([m.float(), (i + 1000).float()], 1)}
        out.append(Case(f'sgl-softcap_argmax-r{R}v{V}', sra._softcap_row_argmax_kernel, None, None, make, ['o'],
                        launch=launch, ref=ref, cmp=('tol', 2 ** -18, 2 ** -18),
                        note='PyTorch reference (float64): the kernel calls libdevice.tanh, which the interpreter does '
                             'not provide; the maximum within 2^-18 relative, its index exact'))

    for (T, H, D, bs, nb, dt, swa) in [(300, 8, 128, 16, 40, torch.bfloat16, False), (1, 32, 64, 32, 4, torch.float16, False),
                                       (2048, 2, 256, 64, 40, torch.float16, True)]:
        def make(g, T=T, H=H, D=D, bs=bs, nb=nb, dt=dt, swa=swa):
            slots = torch.randperm(nb * bs, generator=g)[:T].to(torch.int64)
            t = {'k': gen(dt, (T, H, D), 'rnd', g), 'v': gen(dt, (T, H, D), 'rnd', g), 'slots': slots,
                 'kc': sentinel(dt, (nb, bs, H, D)), 'vc': sentinel(dt, (nb, bs, H, D))}
            if swa:
                t['swa'] = torch.randperm(nb * bs, generator=g).to(torch.int64)
            return t

        def launch(t, interpreted):
            co.launch_reshape_and_cache_flash(t['k'], t['v'], t['kc'], t['vc'], t['slots'], t.get('swa'))
        out.append(Case(f'sgl-reshape_and_cache-{SHORT[dt]}-t{T}h{H}d{D}b{bs}' + ('-swa' if swa else ''),
                        co.reshape_and_cache_flash, None, None, make, ['kc', 'vc'], launch=launch))

    for (npages, pbytes, count, wide) in [(64, 16 * 8 * 128 * 2, 20, True), (300, 4104, 77, False)]:
        def make(g, npages=npages, pbytes=pbytes, count=count):
            perm = torch.randperm(npages, generator=g)
            return {'raw': torch.randint(0, 256, (npages * pbytes,), generator=g, dtype=torch.int32).to(torch.uint8),
                    'src': perm[:count].to(torch.int64), 'dst': perm[count:2 * count].to(torch.int64)}

        def launch(t, interpreted, npages=npages, pbytes=pbytes, wide=wide):
            words = t['raw'][: npages * pbytes].view(torch.int64 if wide else torch.uint8)
            pw = pbytes // (8 if wide else 1)
            cp._copy_pages_kernel[(t['src'].numel(), triton.cdiv(pw, 2048))](
                words, t['src'], t['dst'], pw, SRC_STRIDE=t['src'].stride(0), DST_STRIDE=t['dst'].stride(0), BLOCK=2048)
        out.append(Case(f'sgl-copy_pages-n{npages}b{pbytes}c{count}', cp._copy_pages_kernel, None, None, make, ['raw'],
                        launch=launch, note='launched as copy_pages launches it on a CUDA tensor'))

    for (L, slots, row_bytes, n) in [(8, 4096, 8 * 128 * 2, 37), (32, 1000, 576 * 2, 100)]:
        def make(g, L=L, slots=slots, row_bytes=row_bytes, n=n):
            t = {f'l{i}': torch.randint(0, 256, (slots, row_bytes), generator=g, dtype=torch.int32).to(torch.uint8)
                 for i in range(L)}
            perm = torch.randperm(slots, generator=g)
            t['src'] = perm[:n].to(torch.int64)
            t['tgt'] = perm[n:2 * n].to(torch.int64)
            return t

        def launch(t, interpreted, L=L, row_bytes=row_bytes, n=n):
            dev = t['src'].device
            ptrs = torch.tensor([t[f'l{i}'].data_ptr() for i in range(L)], dtype=torch.uint64, device=dev)
            strides = torch.full((L,), row_bytes, dtype=torch.int64, device=dev)
            cm.copy_all_layer_kv_cache_tiled[(L, triton.cdiv(row_bytes, 512))](
                ptrs, strides, t['tgt'], t['src'], n, triton.next_power_of_2(n), BYTES_PER_TILE=512, num_warps=4,
                num_stages=2)
        out.append(Case(f'sgl-copy_all_layer_kv-l{L}s{slots}b{row_bytes}n{n}', cm.copy_all_layer_kv_cache_tiled, None,
                        None, make, [f'l{i}' for i in range(L)], launch=launch,
                        note='launched as copy_all_layer_kv_cache_func launches it on CUDA (table of pointers)'))

    for (N, W, off, sz, offs) in [(5000, 128, 100, 3000, 7), (300, 4096, 0, 300, 0)]:
        def make(g, N=N, W=W):
            return {'src': gen(torch.float16, (N, W), 'rnd', g), 'dst': sentinel(torch.float16, (N + 50, W))}

        def launch(t, interpreted, off=off, sz=sz, offs=offs):
            dev = t['src'].device
            mc.memcpy_triton(t['dst'], t['src'], 0, torch.tensor([off], device=dev), torch.tensor([sz], device=dev),
                             bool(offs))
        out.append(Case(f'sgl-memcpy-n{N}w{W}o{off}s{sz}', mc.memcpy_triton_kernel, None, None, make, ['dst'],
                        launch=launch))

    for i, (T, D, dt) in enumerate([(1, 8192, torch.float16), (300, 4096, torch.bfloat16), (64, 7168, torch.float32),
                                    (33, 16384, torch.bfloat16)]):
        def make(g, T=T, D=D, dt=dt):
            return {'x': gen(dt, (T, D), 'rnd', g), 'r': gen(dt, (T, D), 'rnd', g), 'w1': gen(dt, (D,), 'rnd', g),
                    'w2': gen(dt, (D,), 'rnd', g), 'y': sentinel(dt, (T, D)), 'o': sentinel(dt, (T, D)),
                    'm': sentinel(dt, (T, D))}

        def launch(t, interpreted):
            t['y'].copy_(fr.fused_rmsnorm(t['x'], t['w1'], 1e-6))
            o, m = fr.fused_dual_residual_rmsnorm(t['x'], t['r'], t['w1'], t['w2'], 1e-6)
            t['o'].copy_(o)
            t['m'].copy_(m)
        # bfloat16: the dual-residual kernel rounds the residual sum to bfloat16 before normalising it
        # (the reference on float32 copies does not): within 2^-6
        out.append(Case(f'sgl-rmsnorm-{SHORT[dt]}-t{T}d{D}', fr.fused_rmsnorm_kernel, None, None, make, ['y', 'o', 'm'],
                        launch=launch, cmp=tol_of(dt, f16=2 ** -9, bf16=2 ** -6, f32=2 ** -18),
                        ref='interp32' if dt == torch.bfloat16 else 'interp'))
    for (T, D, dt) in [(300, 4096, torch.float32), (5, 2880, torch.bfloat16)]:
        def make(g, T=T, D=D, dt=dt):
            return {'x': gen(dt, (T, D), 'rnd', g), 'w': gen(dt, (D,), 'rnd', g), 'y': sentinel(dt, (T, D))}

        def launch(t, interpreted, D=D):
            r32._rmsnorm_fp32_kernel[(t['x'].shape[0],)](t['x'], t['w'], t['y'], D, 1e-6, triton.next_power_of_2(D),
                                                        num_warps=4, enable_fp_fusion=False)
        out.append(Case(f'sgl-rmsnorm_fp32-{SHORT[dt]}-t{T}d{D}', r32._rmsnorm_fp32_kernel, None, None, make, ['y'],
                        launch=launch, cmp=tol_of(dt, bf16=2 ** -7, f32=2 ** -20),
                        ref='interp32' if dt == torch.bfloat16 else 'interp',
                        note='launched as rmsnorm_fp32 launches it on CUDA (it asserts a CUDA tensor)'))
    for (T, I, dt) in [(300, 4096, torch.float16), (1, 14336, torch.bfloat16), (77, 2560, torch.float32)]:
        def make(g, T=T, I=I, dt=dt):
            return {'x': gen(dt, (T, 2 * I), 'rnd', g), 'y': sentinel(dt, (T, I))}

        def launch(t, interpreted):
            gm.gelu_and_mul_triton(t['x'], out=t['y'])
        out.append(Case(f'sgl-gelu_and_mul-{SHORT[dt]}-t{T}i{I}', gm.gelu_and_mul_kernel, None, None, make, ['y'],
                        launch=launch, cmp=tol_of(dt, f16=2 ** -9, bf16=2 ** -7, f32=2 ** -18),
                        ref='interp32' if dt == torch.bfloat16 else 'interp'))
    return out


def sgl_linear_cases():
    """SGLang's copies of the linear-attention and state-space kernels: the gated delta rule of
    Qwen3-Next (chunk_gated_delta_rule_fwd: chunk_local_cumsum, the fused kkt/solve/w-u kernel,
    chunk_delta_h, chunk_o; fused_recurrent_gated_delta_rule), Mamba2's SSD chain through
    mamba_chunk_scan_combined, selective_state_update, causal_conv1d_update. Random data: within
    2^-7 (float16/bfloat16 outputs) or 2^-10 (float32)."""
    sgl_shims()
    from sglang.kernels.ops.attention.fla import chunk as fc, fused_recurrent as frc
    from sglang.kernels.ops.attention.fla import chunk_delta_h, chunk_o, chunk_fwd, cumsum, l2norm, wy_fast
    from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
    from sglang.kernels.ops.mamba.triton_ops import ssd_combined as sc, mamba_ssm as ms
    from sglang.kernels.ops.mamba.triton_ops import ssd_bmm, ssd_chunk_scan, ssd_chunk_state, ssd_state_passing
    from sglang.kernels.ops.mamba import causal_conv1d_triton as cc
    fla_aliases('sglang.kernels.ops.attention.fla')
    for m in (fc, frc, chunk_delta_h, chunk_o, chunk_fwd, cumsum, l2norm, wy_fast, sc, ms, ssd_bmm, ssd_chunk_scan,
              ssd_chunk_state, ssd_state_passing, cc):
        pin_module(m)
    out = []
    for i, (lens, H, K, dt) in enumerate([([100, 1, 200], 8, 128, torch.float16), ([64, 65, 63, 17], 16, 64, torch.float32),
                                          ([700], 4, 128, torch.float16)]):
        def make(g, lens=lens, H=H, K=K, dt=dt):
            T = sum(lens)
            slots = len(lens) + 3
            q = torch.nn.functional.normalize(gen(torch.float32, (1, T, H, K), 'rnd', g), dim=-1)
            k = torch.nn.functional.normalize(gen(torch.float32, (1, T, H, K), 'rnd', g), dim=-1)
            return {'q': q.to(dt), 'k': k.to(dt), 'v': gen(dt, (1, T, H, K), 'rnd', g),
                    'g': torch.nn.functional.logsigmoid(gen(torch.float32, (1, T, H), 'rnd', g) + 3),
                    'beta': torch.rand(1, T, H, generator=g).float().to(dt), 'cu': cu_lens(lens).to(torch.int64),
                    'state': gen(torch.float32, (slots, H, K, K), 'rnd', g) * 0.1,
                    'sidx': torch.randperm(slots, generator=g)[:len(lens)].to(torch.int32),
                    'o': sentinel(dt, (1, T, H, K))}

        def launch(t, interpreted):
            fla_aliases('sglang.kernels.ops.attention.fla')
            cu = t['cu']
            ci = prepare_chunk_indices(cu, 64)
            _, o, *_ = fc.chunk_gated_delta_rule_fwd(t['q'], t['k'], t['v'], t['g'], t['beta'], t['q'].shape[-1] ** -0.5,
                                                     t['state'], t['sidx'], cu_seqlens=cu, chunk_indices=ci,
                                                     inplace_update=True)
            t['o'].copy_(o)
        ot = ('tol', 2 ** -7, 2 ** -7) if dt != torch.float32 else ('tol', 2 ** -10, 2 ** -10)
        out.append(Case(f'sgl-gdn_chunk-{SHORT[dt]}-s{len(lens)}t{sum(lens)}h{H}k{K}', chunk_delta_h.__dict__.get(
            'chunk_gated_delta_rule_fwd_kernel_h_blockdim64'), None, None, make, ['o', 'state'], launch=launch, cmp=ot))
    for (B, H, HV, K, dt) in [(8, 16, 32, 128, torch.float16), (3, 4, 4, 64, torch.float32)]:
        def make(g, B=B, H=H, HV=HV, K=K, dt=dt):
            return {'q': gen(dt, (1, B, H, K), 'rnd', g), 'k': gen(dt, (1, B, H, K), 'rnd', g),
                    'v': gen(dt, (1, B, HV, K), 'rnd', g),
                    'g': torch.nn.functional.logsigmoid(gen(torch.float32, (1, B, HV), 'rnd', g) + 3),
                    'beta': torch.rand(1, B, HV, generator=g).float().to(dt),
                    'h0': gen(torch.float32, (B, HV, K, K), 'rnd', g) * 0.1,
                    'cu': torch.arange(B + 1, dtype=torch.int64), 'o': sentinel(dt, (1, B, HV, K)),
                    'ht': sentinel(torch.float32, (B, HV, K, K))}

        def launch(t, interpreted):
            fla_aliases('sglang.kernels.ops.attention.fla')
            o, ht = frc.fused_recurrent_gated_delta_rule(t['q'], t['k'], t['v'], t['g'], t['beta'], initial_state=t['h0'],
                                                         output_final_state=True, cu_seqlens=t['cu'],
                                                         use_qk_l2norm_in_kernel=True)
            t['o'].copy_(o)
            t['ht'].copy_(ht)
        ot = ('tol', 2 ** -7, 2 ** -7) if dt != torch.float32 else ('tol', 2 ** -10, 2 ** -10)
        out.append(Case(f'sgl-gdn_recurrent-{SHORT[dt]}-b{B}h{H}hv{HV}k{K}', frc.fused_recurrent_gated_delta_rule_fwd_kernel,
                        None, None, make, ['o', 'ht'], launch=launch, cmp=ot))

    for (B, T, H, P, G, N, chunk, dt) in [(2, 300, 16, 64, 2, 128, 256, torch.float16), (1, 1000, 8, 64, 1, 64, 128, torch.float32)]:
        def make(g, B=B, T=T, H=H, P=P, G=G, N=N, dt=dt):
            return {'x': gen(dt, (B, T, H, P), 'rnd', g), 'dt': (gen(torch.float32, (B, T, H), 'rnd', g) * 0.5 - 1).to(dt),
                    'A': -torch.rand(H, generator=g).float() * 2 - 0.1, 'B': gen(dt, (B, T, G, N), 'rnd', g),
                    'C': gen(dt, (B, T, G, N), 'rnd', g), 'D': gen(torch.float32, (H,), 'rnd', g),
                    'z': gen(dt, (B, T, H, P), 'rnd', g), 'dt_bias': gen(torch.float32, (H,), 'rnd', g) * 0.1,
                    'out': sentinel(dt, (B, T, H, P)), 'fs': sentinel(torch.float32, (B, H, P, N))}

        def launch(t, interpreted, chunk=chunk):
            r = sc.mamba_chunk_scan_combined(t['x'], t['dt'], t['A'], t['B'], t['C'], chunk, D=t['D'], z=t['z'],
                                             dt_bias=t['dt_bias'], dt_softplus=True, out=t['out'],
                                             return_final_states=True)
            t['fs'].copy_(r[-1] if isinstance(r, (tuple, list)) else r)
        ot = ('tol', 2 ** -7, 2 ** -7) if dt != torch.float32 else ('tol', 2 ** -10, 2 ** -10)
        out.append(Case(f'sgl-mamba2_ssd-{SHORT[dt]}-b{B}t{T}h{H}p{P}g{G}n{N}c{chunk}', ssd_chunk_scan.__dict__.get(
            '_chunk_scan_fwd_kernel'), None, None, make, ['out', 'fs'], launch=launch, cmp=ot))

    for (B, H, P, G, N, dt) in [(8, 16, 64, 8, 128, torch.float16), (32, 8, 64, 2, 64, torch.bfloat16)]:
        def make(g, B=B, H=H, P=P, G=G, N=N, dt=dt):
            slots = B * 2 + 3
            idx = torch.randperm(slots, generator=g)[:B].to(torch.int32)
            idx[1] = -1
            return {'state': gen(dt, (slots, H, P, N), 'rnd', g), 'x': gen(dt, (B, H, P), 'rnd', g),
                    'dt': gen(dt, (B, H), 'rnd', g), 'A': -torch.rand(H, generator=g) - 0.1,
                    'B': gen(dt, (B, G, N), 'rnd', g), 'C': gen(dt, (B, G, N), 'rnd', g), 'D': gen(torch.float32, (H,), 'rnd', g),
                    'z': gen(dt, (B, H, P), 'rnd', g), 'dt_bias': gen(torch.float32, (H,), 'rnd', g) * 0.1, 'idx': idx,
                    'out': sentinel(dt, (B, H, P))}

        def launch(t, interpreted, P=P, N=N):
            A = t['A'][:, None, None].expand(-1, P, N)
            D = t['D'][:, None].expand(-1, P)
            dtb = t['dt_bias'][:, None].expand(-1, P)
            dtt = t['dt'][:, :, None].expand(-1, -1, P)
            ms.selective_state_update(t['state'], t['x'], dtt, A, t['B'], t['C'], D=D, z=t['z'], dt_bias=dtb,
                                      dt_softplus=True, state_batch_indices=t['idx'], pad_slot_id=-1, out=t['out'])
        out.append(Case(f'sgl-selective_state_update-{SHORT[dt]}-b{B}h{H}p{P}g{G}n{N}', ms._selective_scan_update_kernel,
                        None, None, make, ['out', 'state'], launch=launch, cmp=('tol', 2 ** -7, 2 ** -7),
                        ref='interp32' if dt == torch.bfloat16 else 'interp'))

    for (B, D, W, dt) in [(8, 4096, 4, torch.float16), (64, 2048, 4, torch.bfloat16)]:
        def make(g, B=B, D=D, W=W, dt=dt):
            nslots = B + 7
            return {'x': gen(dt, (B, D), 'int', g), 'w': gen(dt, (D, W), 'small', g), 'b': gen(dt, (D,), 'small', g),
                    'state': gen(dt, (nslots, D, W - 1), 'small', g),
                    'idx': torch.randperm(nslots, generator=g)[:B].to(torch.int32)}

        def launch(t, interpreted):
            t['x'].copy_(cc.causal_conv1d_update(t['x'], t['state'], t['w'], t['b'], activation=None,
                                                 conv_state_indices=t['idx']))
        out.append(Case(f'sgl-causal_conv1d_update-{SHORT[dt]}-b{B}d{D}w{W}', cc._causal_conv1d_update_kernel, None, None,
                        make, ['x', 'state'], launch=launch, ref='interp32' if dt == torch.bfloat16 else 'interp'))
    return out


def sgl_lora_ref(R, Nq, Nkv):
    """SGLang's dense LoRA with PyTorch on the CPU: shrink (stored in the input type), then the qkv
    expand onto a base output and the plain expand, each product scaled and rounded to the input
    type before it is added to the output (as _qkv_lora_b_kernel and _sgemm_lora_b_kernel do)."""
    def ref(t):
        dt = t['x'].dtype
        x, base = t['x'].float(), t['base'].float()
        T = x.shape[0]
        mid = torch.zeros(T, 3 * R)
        y = torch.zeros(T, Nq + 2 * Nkv)
        y1 = torch.zeros(T, Nq)
        offs = [0, Nq, Nq + Nkv, Nq + 2 * Nkv]
        ip = t['indptr'].tolist()
        for i in range(len(ip) - 1):
            a = int(t['widx'][i])
            sc = float(t['scal'][a])
            rows = slice(ip[i], ip[i + 1])
            m = (x[rows] @ t['wa'][a].float().t()).to(dt).float()
            mid[rows] = m
            for j in range(3):
                p = (m[:, j * R:(j + 1) * R] @ t['wb'][a, offs[j]:offs[j + 1]].float().t() * sc).to(dt).float()
                y[rows, offs[j]:offs[j + 1]] = (p + base[rows, offs[j]:offs[j + 1]]).to(dt).float()
            # the plain expand reads the adapter's rank (R) columns of its input, of the 3 R it is given
            y1[rows] = (m[:, :R] @ t['wb1'][a, :, :R].float().t() * sc).to(dt).float()
        return {'mid': mid.to(dt), 'y': y.to(dt), 'y1': y1.to(dt)}
    return ref


def sgl_quant_lora_spec_cases():
    """SGLang's int8 quantization (per-token, per-token-group), its LoRA kernels of dense layers
    (_sgemm_lora_a_kernel, _sgemm_lora_b_kernel, _qkv_lora_b_kernel through their launchers, with
    segments of several adapters) and two kernels of EAGLE speculative decoding (bonus tokens,
    accepted cache locations). Small integers and powers of two: exact."""
    sgl_shims()
    import types
    from sglang.kernels.ops.quantization import int8_kernel as i8
    from sglang.kernels.ops.lora.dense import sgemm_lora_a as la, sgemm_lora_b as lb, qkv_lora_b as qb
    from sglang.kernels.ops.speculative import eagle as eg
    out = []
    for (M, N, G, dt) in [(1, 4096, 128, torch.float16), (300, 7168, 128, torch.bfloat16), (17, 5120, 5120, torch.float32)]:
        def make(g, M=M, N=N, G=G, dt=dt):
            return {'x': quant_rows(g, M, N, G, dt), 'q': sentinel(torch.int8, (M, N)),
                    's': sentinel(torch.float32, (M, N // G))}

        def launch(t, interpreted, N=N, G=G):
            if G == N:
                q, s_ = i8.per_token_quant_int8(t['x'])
            else:
                q, s_ = i8.per_token_group_quant_int8(t['x'], G)
            t['q'].copy_(q)
            t['s'].copy_(s_.reshape(t['s'].shape))
        out.append(Case(f'sgl-int8_quant-{SHORT[dt]}-m{M}n{N}g{G}', i8._per_token_group_quant_int8 if G != N else
                        i8._per_token_quant_int8, None, None, make, ['q', 's'], launch=launch,
                        ref=int8_row_ref if G == N else ('interp32' if dt == torch.bfloat16 else 'interp'),
                        note='PyTorch reference: the kernel calls tl.extra.cuda.libdevice.round' if G == N else ''))

    for i, (segs, L, K, R, Nq, Nkv, dt) in enumerate([([30, 1, 77, 5], 4, 4096, 16, 4096, 1024, torch.float16),
                                                      ([1] * 32, 8, 2048, 64, 2048, 256, torch.bfloat16)]):
        def make(g, segs=segs, L=L, K=K, R=R, Nq=Nq, Nkv=Nkv, dt=dt):
            T = sum(segs)
            return {'x': gen(dt, (T, K), 'small', g), 'wa': gen(dt, (L, 3 * R, K), 'small', g),
                    'wb': gen(dt, (L, Nq + 2 * Nkv, R), 'small', g), 'base': gen(dt, (T, Nq + 2 * Nkv), 'small', g),
                    'wb1': gen(dt, (L, Nq, 3 * R), 'small', g),
                    'seg': torch.tensor(segs, dtype=torch.int32), 'indptr': cu_lens(segs),
                    'widx': torch.randint(0, L, (len(segs),), generator=g, dtype=torch.int32),
                    'ranks': torch.full((L,), R, dtype=torch.int32), 'scal': dyadic((L,), g, (0.5, 1.0, 2.0)).float(),
                    'mid': sentinel(dt, (T, 3 * R)), 'y': sentinel(dt, (T, Nq + 2 * Nkv)), 'y1': sentinel(dt, (T, Nq))}

        def launch(t, interpreted, segs=segs, R=R, Nq=Nq, Nkv=Nkv):
            bi = types.SimpleNamespace(use_cuda_graph=False, bs=len(segs), num_segments=len(segs), seg_indptr=t['indptr'],
                                       weight_indices=t['widx'], lora_ranks=t['ranks'], scalings=t['scal'],
                                       max_len=max(segs), seg_lens=t['seg'], permutation=None, expected_tokens=None,
                                       has_active_lora=True, is_prefill=True, req_seg_indptr=None)
            mid = la.sgemm_lora_a_fwd(t['x'], t['wa'], bi, stack_num=3)
            t['mid'].copy_(mid)
            off = torch.tensor([0, Nq, Nq + Nkv, Nq + 2 * Nkv], dtype=torch.int32, device=mid.device)
            t['y'].copy_(qb.qkv_lora_b_fwd(mid, t['wb'], bi, off, max(Nq, Nkv), base_output=t['base'].clone()))
            t['y1'].copy_(lb.sgemm_lora_b_fwd(mid, t['wb1'], bi))
        bf = dt == torch.bfloat16
        out.append(Case(f'sgl-lora-{SHORT[dt]}-s{len(segs)}t{sum(segs)}l{L}k{K}r{R}', la._sgemm_lora_a_kernel, None, None,
                        make, ['mid', 'y', 'y1'], launch=launch, ref=sgl_lora_ref(R, Nq, Nkv) if bf else 'interp',
                        note='bfloat16: PyTorch reference with the roundings the kernels make (the shrink output stored '
                             'in bfloat16; the scaled product rounded to bfloat16, then added to the base output in '
                             'bfloat16); the interpreter on float32 copies would make none of them' if bf else ''))

    for (B, S) in [(8, 4), (64, 6)]:
        def make(g, B=B, S=S):
            acc = torch.randint(1, S + 1, (B,), generator=g, dtype=torch.int32)
            aidx = torch.full((B * S,), -1, dtype=torch.int32)
            for b in range(B):
                aidx[b * S:b * S + int(acc[b])] = torch.arange(b * S, b * S + int(acc[b]), dtype=torch.int32)
            return {'tok': torch.randint(0, 150000, (B * S,), generator=g, dtype=torch.int32), 'acc': acc,
                    'bonus': sentinel(torch.int32, (B,)), 'aidx': aidx,
                    'loc': torch.randint(0, 2 ** 20, (B * S,), generator=g, dtype=torch.int64),
                    'aloc': sentinel(torch.int64, (B * S,))}

        def launch(t, interpreted, B=B, S=S):
            eg.fill_bonus_tokens[(B,)](t['tok'], t['acc'], t['bonus'], S)
            eg.fill_accept_out_cache_loc[(B * S,)](t['aidx'], t['loc'], t['aloc'], triton.next_power_of_2(B * S))
        out.append(Case(f'sgl-eagle_accept-b{B}s{S}', eg.fill_accept_out_cache_loc, None, None, make, ['bonus', 'aloc'],
                        launch=launch, note='launched as the launchers launch them on CUDA'))
    return out


def sgl_router_gate_cases():
    """SGLang's MoE routers (fused_moe_router_cudacore_kernel and fused_moe_router_tensorcore_kernel:
    the router product, softcapping, top-k, with a correction bias) and the attention-output gate of
    Qwen3.5 (_fused_sigmoid_mul_kernel with a strided 3-D gate). Random data: weights within
    2^-12, expert ids exact; the gate within 2^-9 (float16) / 2^-7 (bfloat16)."""
    sgl_shims()
    from sglang.kernels.ops.moe import router as rt
    from sglang.kernels.ops.elementwise import elementwise as el
    out = []
    for (T, H, E, topk, cap, bias, tc) in [(64, 6144, 8, 2, 30.0, False, False), (1, 4096, 128, 8, 30.0, True, False),
                                           (300, 2048, 16, 2, 50.0, False, True), (33, 4096, 64, 1, 30.0, True, True)]:
        def make(g, T=T, H=H, E=E, topk=topk, bias=bias):
            t = {'x': gen(torch.bfloat16, (T, H), 'rnd', g), 'w': (gen(torch.float32, (E, H), 'rnd', g) / 8).to(torch.bfloat16),
                 'tw': sentinel(torch.float32, (T, topk)), 'ti': sentinel(torch.int32, (T, topk))}
            if bias:
                t['bias'] = gen(torch.float32, (E,), 'rnd', g) * 0.1
            return t

        def launch(t, interpreted, topk=topk, cap=cap, tc=tc, E=E):
            if tc:
                w, i = rt.fused_moe_router_tensorcore(t['x'], t['w'], topk, cap, 32, max(16, triton.next_power_of_2(E)), 64,
                                                      correction_bias=t.get('bias'))
            else:
                w, i = rt.fused_moe_router_cudacore(t['x'], t['w'], topk, cap, correction_bias=t.get('bias'))
            t['tw'].copy_(w)
            t['ti'].copy_(i)
        out.append(Case(f'sgl-moe_router-{"tc" if tc else "cc"}-t{T}h{H}e{E}k{topk}' + ('-bias' if bias else ''),
                        rt.fused_moe_router_tensorcore_kernel if tc else rt.fused_moe_router_cudacore_kernel, None,
                        None, make, ['tw', 'ti'], launch=launch, ref='interp32', cmp=('tol', 2 ** -12, 2 ** -12)))
    for (T, Hh, D, dt) in [(300, 32, 128, torch.bfloat16), (1, 16, 256, torch.float16), (2048, 8, 128, torch.float16)]:
        def make(g, T=T, Hh=Hh, D=D, dt=dt):
            # the gate is a strided view: every other head of a wider projection
            return {'o': gen(dt, (T, Hh * D), 'rnd', g), 'gate': gen(dt, (T, 2 * Hh, D), 'rnd', g),
                    'y': sentinel(dt, (T, Hh * D))}

        def launch(t, interpreted):
            t['y'].copy_(el.fused_sigmoid_mul(t['o'], t['gate'][:, ::2, :]))
        out.append(Case(f'sgl-sigmoid_mul-{SHORT[dt]}-t{T}h{Hh}d{D}', el._fused_sigmoid_mul_kernel, None, None, make, ['y'],
                        launch=launch, cmp=tol_of(dt, f16=2 ** -9, bf16=2 ** -7),
                        ref='interp32' if dt == torch.bfloat16 else 'interp'))
    return out


def sgl_triton_kernels_cases():
    """The triton_kernels package SGLang ships (the dense path of its matmul, which the gpt-oss MoE
    path builds on). It asks the device for its target when it chooses a configuration, which the
    interpreter cannot answer: the reference is PyTorch on the CPU (small integers: exact)."""
    try:
        import triton_kernels.matmul as tkm
    except Exception:
        return []
    out = []
    for (M, K, N, dt) in [(64, 256, 128, torch.float16), (300, 2048, 1024, torch.bfloat16), (1, 4096, 4096, torch.float16)]:
        def make(g, M=M, K=K, N=N, dt=dt):
            return {'x': gen(dt, (M, K), 'small', g), 'w': gen(dt, (K, N), 'small', g), 'y': sentinel(dt, (M, N))}

        def launch(t, interpreted):
            t['y'].copy_(tkm.matmul(t['x'], t['w'], None))

        def ref(t):
            return {'y': (t['x'].float() @ t['w'].float()).to(t['y'].dtype)}
        out.append(Case(f'sgl-tk_matmul-{SHORT[dt]}-m{M}k{K}n{N}', None, None, None, make, ['y'], launch=launch, ref=ref,
                        note='triton_kernels.matmul.matmul; PyTorch reference'))
    return out


SGL_FAMILIES = [sgl_attention_cases, sgl_moe_cases, sgl_misc_cases, sgl_linear_cases, sgl_quant_lora_spec_cases,
                sgl_router_gate_cases, sgl_triton_kernels_cases]

# ------------------------------------------------------------------------------------------------


def _have(name):
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


# The cases of tests/gpu/triton.sh's quick level: one or two of each family, with a reference the
# interpreter makes in seconds.
QUICK = {
    'vllm-fused_moe-f16-m64e8k2-2048x1024-64x64x32w4s3', 'vllm-fused_moe_wna16-int4-zp-m19e8k2-768x1024-g128',
    'vllm-unified_attn-2d-f16-s4q143-h16kv2d64p32', 'vllm-unified_attn-3d-bf16-s6q6-h28kv4d128p16',
    'vllm-decode_attn-mla-f16-b6-h16kv1d576-p64s4', 'vllm-reshape_and_cache_flash-bf16-t300h8d128b16',
    'vllm-penalties-t64r64v152064', 'vllm-topk_topp-b16v151936-k40-p0.9', 'vllm-gdn_chunk-f16-s3t301h8k128-h0',
    'vllm-input_batch-r64n40s3l3000c512', 'vllm-scaled_mm-i8-m100k2048n1536-tok-bias',
    'vllm-bi_matmul-bf16-m300k1024n2048-bias',
    'sgl-decode_attn-gqa-f16-b8-h32kv8d128-s8', 'sgl-extend_attn-f16-s4q212p644-h32kv8d128',
    'sgl-fused_moe-f16-m200e16k4-520x384-64x128x32w8s3-rw-ep', 'sgl-moe_align_small-m8e128k8b64',
    'sgl-reshape_and_cache-bf16-t300h8d128b16', 'sgl-rmsnorm-bf16-t33d16384', 'sgl-gdn_chunk-f16-s3t301h8k128',
    'sgl-lora-f16-s4t113l4k4096r16',
}


def cases(level='full'):
    out = []
    fams = []
    if _have('vllm'):
        fams += VLLM_FAMILIES
    if _have('sglang'):
        fams += SGL_FAMILIES
    for f in fams:
        out += f()
    names = [c.name for c in out]
    assert len(names) == len(set(names)), 'two cases of one name'
    assert all(n.startswith(('vllm-', 'sgl-')) for n in names)
    for c in out:
        c.quick = c.name in QUICK
    if level == 'quick':
        out = [c for c in out if c.quick]
    return out
