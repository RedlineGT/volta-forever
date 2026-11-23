#!/usr/bin/env python3
"""Test the _topk_topp_kernel against a CPU reference.

The kernel applies combined top-k and top-p filtering to logits in-place:
  - top-k: keep only the k highest logits, mask the rest to -inf
  - top-p: among the top-k survivors, keep only the smallest set whose
    softmax probabilities sum to at least p, mask the rest to -inf
  - When both are active, top-k is applied first, then top-p within the survivors.

This script compares the Triton kernel (through the layer on a V100) with the SAME kernel
run on the CPU by Triton's interpreter (TRITON_INTERPRET=1, in a child process), for several
batch sizes, vocabulary sizes, k and p values: the two must keep the same positions with the
same values. A second computation on the CPU, written here with a sort and a cumulative sum,
is printed beside it and decides nothing: it is another algorithm than the kernel's, and at
the cut of a top-p it keeps a few positions more or fewer than the kernel does on any
machine (2 of 32,000 at p=0.99; 7 of 128,000 at p=0.9 over four rows).

Environment:
  VF_PROXY must be set (the proxy's lib directory).
  PYTHONPATH must include vllm and torch sites.
Prints PASS or FAIL per case, and a summary; the exit status is 1 when a case fails. The
comparison is checked first, on tensors made for it (`--comparison`: that alone, no GPU needed).
"""
import importlib.util, os, subprocess, sys, tempfile, torch, math

# --- CPU reference ---

def topk_topp_cpu(logits, k=None, p=None, mask_value=float('-inf')):
    """CPU reference for combined top-k / top-p filtering.

    Args:
        logits: (batch_size, vocab_size) float32 tensor, modified in-place
        k: (batch_size,) int32 tensor or None (top-k disabled)
        p: (batch_size,) float32 tensor or None (top-p disabled)
        mask_value: value to set for masked-out tokens
    Returns:
        logits (modified in-place)
    """
    batch_size, vocab_size = logits.shape
    for row in range(batch_size):
        row_logits = logits[row]

        # Count finite (non -inf) logits
        finite_mask = row_logits > float('-inf')
        num_finite = finite_mask.sum().item()

        if k is not None:
            ki = k[row].item()
            if ki < vocab_size and num_finite > ki:
                # top-k: keep only the top ki logits
                # Sort descending, find the ki-th largest value
                sorted_vals, sorted_idx = torch.sort(row_logits, descending=True)
                # The pivot is the ki-th value (0-indexed: ki-1)
                # Keep logits strictly greater than pivot, plus enough duplicates at pivot
                pivot = sorted_vals[ki - 1].item() if ki > 0 else float('inf')

                # Find the boundary: how many are strictly above pivot
                above = (row_logits > pivot).sum().item()
                # Need to keep exactly ki total: keep 'above' that are strictly above,
                # plus (ki - above) of those equal to pivot
                need_at_pivot = ki - above

                keep_mask = row_logits > pivot
                if need_at_pivot > 0:
                    # Keep the first need_at_pivot entries that equal pivot
                    at_pivot = (torch.abs(row_logits - pivot) < 1e-9)
                    cumsum = torch.cumsum(at_pivot.int(), dim=0)
                    keep_at_pivot = at_pivot & (cumsum <= need_at_pivot)
                    keep_mask = keep_mask | keep_at_pivot

                row_logits[~keep_mask] = mask_value

                if p is not None:
                    pi = p[row].item()
                    if pi < 1.0:
                        _apply_topp_after_topk(row_logits, pi, mask_value)
            elif p is not None:
                # k >= vocab_size or num_finite <= k: top-k is a no-op
                pi = p[row].item()
                if pi < 1.0:
                    _apply_topp_standalone(row_logits, pi, mask_value)
        elif p is not None:
            pi = p[row].item()
            if pi < 1.0:
                _apply_topp_standalone(row_logits, pi, mask_value)

    return logits


def _apply_topp_after_topk(row_logits, p, mask_value):
    """Apply top-p within the surviving (non-masked) logits."""
    alive = row_logits > float('-inf')
    if alive.sum().item() == 0:
        return
    alive_logits = row_logits[alive]
    probs = torch.softmax(alive_logits.float(), dim=0)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)

    # Find the smallest set with cumsum >= p
    # Keep all tokens whose cumsum (up to and including that token) is needed
    cutoff_idx = (cumsum >= p).nonzero(as_tuple=True)[0]
    if len(cutoff_idx) == 0:
        return  # keep all
    cutoff = cutoff_idx[0].item()
    # Mask everything after cutoff
    to_remove = sorted_idx[cutoff + 1:]
    alive_indices = alive.nonzero(as_tuple=True)[0]
    for idx in to_remove:
        row_logits[alive_indices[idx]] = mask_value


def _apply_topp_standalone(row_logits, p, mask_value):
    """Apply top-p to all logits (no prior top-k)."""
    finite = row_logits > float('-inf')
    if finite.sum().item() == 0:
        return
    probs = torch.softmax(row_logits.float(), dim=0)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)

    cutoff_idx = (cumsum >= p).nonzero(as_tuple=True)[0]
    if len(cutoff_idx) == 0:
        return
    cutoff = cutoff_idx[0].item()
    to_remove = sorted_idx[cutoff + 1:]
    for idx in to_remove:
        row_logits[idx] = mask_value


# --- Run the Triton kernel ---

def interpreted_kernel():
    """The kernel for Triton's interpreter: its source with one cast removed. The interpreter
    gives the loop variable as a Python int, which has no `.to`; nothing else changes."""
    import vllm.v1.sample.ops.topk_topp_triton as module
    source = open(module.__file__).read()
    assert source.count('row_id.to(tl.int64)') == 1, 'the kernel changed: look at its loop'
    # In a file of its own: the interpreter reads a kernel's source again from its file.
    path = os.path.join(tempfile.mkdtemp(), 'topk_topp_interpreted.py')
    with open(path, 'w') as f:
        f.write(source.replace('row_id.to(tl.int64)', 'row_id'))
    spec = importlib.util.spec_from_file_location('topk_topp_interpreted', path)
    copy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copy)
    return copy._topk_topp_kernel


def run_triton_topk_topp(logits_gpu, k_gpu, p_gpu, mask_value, vocab_size, on_cpu=False):
    """Run the actual vLLM Triton kernel: on the GPU, or on the CPU under the interpreter."""
    if on_cpu:
        _topk_topp_kernel = interpreted_kernel()
    else:
        from vllm.v1.sample.ops.topk_topp_triton import _topk_topp_kernel
    device = logits_gpu.device

    batch_size = logits_gpu.shape[0]

    topk_enabled = k_gpu is not None
    topp_enabled = p_gpu is not None
    k_ptr = k_gpu if k_gpu is not None else logits_gpu
    p_ptr = p_gpu if p_gpu is not None else logits_gpu

    block_size = 8192
    block_size_trunc = 4096
    num_warps = 8
    if on_cpu:
        num_programs = batch_size
    else:
        from vllm.utils.platform_utils import num_compute_units
        num_programs = min(num_compute_units(), batch_size)

    buffer = torch.empty((num_programs, vocab_size), dtype=torch.float32, device=device)

    # Build the lookup tables
    from vllm.v1.sample.ops.topk_topp_triton import (
        _PERCENTILE_TO_STD_TABLE, _NORMAL_CDF_TO_SIGMA_TABLE
    )
    pct_table = torch.tensor(_PERCENTILE_TO_STD_TABLE, dtype=torch.float32, device=device)
    cdf_table = torch.tensor(_NORMAL_CDF_TO_SIGMA_TABLE, dtype=torch.float32, device=device)

    # What vLLM passes on a CUDA device; the same on the CPU, so that both run one kernel.
    split_covers_ponly = topp_enabled and batch_size <= 64

    grid = (num_programs,)
    _topk_topp_kernel[grid](
        logits_gpu,
        logits_gpu.stride(0),
        buffer,
        pct_table,
        cdf_table,
        k_ptr,
        p_ptr,
        batch_size,
        VOCAB_SIZE=vocab_size,
        MASK_VALUE=mask_value,
        BLOCK_SIZE=block_size,
        BLOCK_SIZE_TRUNC=block_size_trunc,
        TOPK_ENABLED=topk_enabled,
        TOPP_ENABLED=topp_enabled,
        SPLIT_COVERS_PONLY=split_covers_ponly,
        num_warps=num_warps,
    )
    if not on_cpu:
        torch.cuda.synchronize()
    return logits_gpu


def compare_results(gpu_logits, cpu_logits, label):
    """Compare GPU and CPU results, the whole tensors: the logits of every case are finite, so
    what the kernel leaves is a logit as it was or the mask value, and a NaN on either side is a
    failure. Returns (pass, message)."""
    alive = int((cpu_logits > float('-inf')).sum().item())
    if torch.equal(gpu_logits, cpu_logits):
        return True, f'{label}: EXACT match ({alive} alive)'
    nans = int(gpu_logits.isnan().sum().item()), int(cpu_logits.isnan().sum().item())
    if any(nans):
        return False, f'{label}: NaN in the result: gpu={nans[0]} cpu={nans[1]}'

    # Both should have the same set of non-masked positions
    gpu_alive = gpu_logits > float('-inf')
    cpu_alive = cpu_logits > float('-inf')

    if not torch.equal(gpu_alive, cpu_alive):
        # Count differences
        gpu_count = gpu_alive.sum().item()
        cpu_count = cpu_alive.sum().item()
        diff_positions = (gpu_alive != cpu_alive).sum().item()
        return False, f'{label}: alive set differs: gpu={gpu_count} cpu={cpu_count} diff_positions={diff_positions}'

    # Where both are alive, the values differ
    both_alive = gpu_alive & cpu_alive
    max_diff = (gpu_logits[both_alive] - cpu_logits[both_alive]).abs().max().item()
    return False, f'{label}: values differ, max_diff={max_diff:.6e} ({both_alive.sum().item()} alive)'


def comparison_checks():
    """What compare_results must refuse, and take: a check that was never seen to fail has not
    been shown to check. (Until 2026-11-23 it compared the values kept alone, and took a NaN for
    a masked logit: [1, NaN] passed against [1, -inf].)"""
    inf, nan = float('inf'), float('nan')
    wrong = []
    for gpu, cpu, same in [
        ([1.0, -inf], [1.0, -inf], True),
        ([-inf, -inf], [-inf, -inf], True),
        ([1.0, nan], [1.0, -inf], False),
        ([nan, -inf], [1.0, -inf], False),
        ([1.0, nan], [1.0, nan], False),
        ([1.0, 2.0], [1.0, -inf], False),
        ([1.0, 2.5], [1.0, 2.0], False),
    ]:
        ok, msg = compare_results(torch.tensor([gpu]), torch.tensor([cpu]), f'{gpu} against {cpu}')
        if ok != same:
            wrong.append(f'{msg} (should {"pass" if same else "fail"})')
    # (Until 2026-11-23 the cases were zipped with the interpreter's results: with one result
    # fewer, 17 cases were compared and PASS said.)
    if paired([torch.zeros(1)] * (len(CASES) - 1)) is not None or paired([torch.zeros(1)] * len(CASES)) is None:
        wrong.append('a reference with one result fewer than the cases is paired with them, or one with all is not')
    return wrong


def paired(on_cpu):
    """[(case, the interpreter's result)], or None when the results are not as many as the cases."""
    if not isinstance(on_cpu, (list, tuple)) or len(on_cpu) != len(CASES):
        return None
    return list(zip(CASES, on_cpu))


# Test cases: (batch_size, vocab_size, k, p, description)
CASES = [
    # top-k only
    (1, 32000, 50, None, 'top-k only, small k'),
    (1, 32000, 1, None, 'top-k=1 (argmax)'),
    (1, 32000, 31999, None, 'top-k nearly all'),
    (4, 32000, 50, None, 'top-k, batch=4'),
    (1, 151936, 50, None, 'top-k, Qwen vocab'),

    # top-p only
    (1, 32000, None, 0.9, 'top-p=0.9'),
    (1, 32000, None, 0.1, 'top-p=0.1 (very selective)'),
    (1, 32000, None, 0.99, 'top-p=0.99 (nearly all)'),
    (1, 32000, None, 0.01, 'top-p near 0'),
    (4, 32000, None, 0.9, 'top-p, batch=4'),

    # combined top-k + top-p
    (1, 32000, 50, 0.9, 'top-k=50 + top-p=0.9'),
    (1, 32000, 50, 0.1, 'top-k=50 + top-p=0.1'),
    (1, 32000, 10, 0.9, 'top-k=10 + top-p=0.9'),
    (4, 32000, 50, 0.9, 'combined, batch=4'),
    (1, 151936, 50, 0.9, 'combined, Qwen vocab'),

    # edge: k disabled (k >= vocab_size), p active.  With top-k enabled and a batch of 64 rows
    # or fewer, the kernel leaves the rows whose k >= VOCAB_SIZE to vLLM's split pipeline
    # (_topp_split_* kernels), which this test does not run: all their logits stay alive.
    (1, 32000, 32000, 0.9, 'k=vocab (split covers, no-op expected)'),
    (1, 32000, 100000, 0.9, 'k>vocab (split covers, no-op expected)'),

    # ties: many equal values, k=50 from 100 equal top values
    (1, 1000, 50, None, 'ties'),
]


def inputs(batch_size, vocab_size, k_val, p_val, desc, seed=42):
    """The logits, k and p of a case, on the CPU."""
    if desc == 'ties':
        logits = torch.zeros(batch_size, vocab_size, dtype=torch.float32)
        logits[0, :100] = 1.0
        logits[0, 100:200] = 0.5
        logits[0, 200:300] = 0.0
    else:
        torch.manual_seed(seed)
        logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    k = None if k_val is None else torch.full((batch_size,), k_val, dtype=torch.int32)
    p = None if p_val is None else torch.full((batch_size,), p_val, dtype=torch.float32)
    return logits, k, p


def kernel_result(case, device):
    """What the kernel leaves of a case's logits, run on `device`; on the CPU, interpreted."""
    batch_size, vocab_size, k_val, p_val, desc = case
    logits, k, p = inputs(*case)
    x = logits.clone().to(device)
    run_triton_topk_topp(x, None if k is None else k.to(device), None if p is None else p.to(device),
                         float('-inf'), vocab_size, on_cpu=device == 'cpu')
    return x.cpu()


def sorted_reference(case):
    """The other computation on the CPU (a sort and a cumulative sum): printed, decides nothing."""
    batch_size, vocab_size, k_val, p_val, desc = case
    logits, k, p = inputs(*case)
    ref = logits.clone()
    if k is not None and p is not None and batch_size <= 64:
        for row in range(batch_size):
            if k[row].item() < vocab_size:
                topk_topp_cpu(ref[row:row + 1], k=k[row:row + 1].clone(), p=p[row:row + 1].clone())
    else:
        topk_topp_cpu(ref, k=None if k is None else k.clone(), p=None if p is None else p.clone())
    return ref


def interpreted(path):
    """Child process (TRITON_INTERPRET=1, no GPU): the kernel's result for every case, saved."""
    torch.save([kernel_result(case, 'cpu') for case in CASES], path)
    return 0


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--interpreted':
        return interpreted(sys.argv[2])
    wrong = comparison_checks()
    for msg in wrong:
        print(f'FAIL: the comparison: {msg}')
    if wrong:
        return 1
    if sys.argv[1:] == ['--comparison']:
        print('PASS the comparison')
        return 0

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'interpreted.pt')
        env = dict(os.environ, TRITON_INTERPRET='1', CUDA_VISIBLE_DEVICES='')
        for name in ('LD_AUDIT', 'LD_PRELOAD'):
            env.pop(name, None)
        child = subprocess.run([sys.executable, '-s', os.path.abspath(__file__), '--interpreted', path],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if child.returncode != 0 or not os.path.exists(path):
            print(child.stdout[-4000:])
            print(f'FAIL: the kernel could not be run on the CPU by Triton\'s interpreter (status {child.returncode})')
            return 1
        on_cpu = torch.load(path)
    pairs = paired(on_cpu)
    if pairs is None:
        print(f'FAIL: the interpreter gave {len(on_cpu) if isinstance(on_cpu, (list, tuple)) else "no list of"} results '
              f'for {len(CASES)} cases')
        return 1

    passed = 0
    failed_cases = []
    for case, cpu_result in pairs:
        batch_size, vocab_size, k_val, p_val, desc = case
        label = f'batch={batch_size} vocab={vocab_size} k={k_val} p={p_val}'
        try:
            gpu_result = kernel_result(case, 'cuda')
        except Exception as e:
            ok, msg = False, f'{label}: GPU error: {e}'
        else:
            ok, msg = compare_results(gpu_result, cpu_result, label)
            if desc == 'ties':
                # Which of the equal values the reference keeps is its own choice: count them.
                other = f'the sorted reference keeps {int((sorted_reference(case) > float("-inf")).sum())}'
            else:
                same, _ = compare_results(gpu_result, sorted_reference(case), label)
                other = 'the sorted reference agrees' if same else (
                    'the sorted reference keeps %d (another algorithm: see the head of this file)'
                    % int((sorted_reference(case) > float('-inf')).sum()))
            msg = f'{msg}; {other}'
        if ok:
            passed += 1
            print(f'  PASS: {desc} -- {msg}')
        else:
            failed_cases.append(msg)
            print(f'  FAIL: {desc} -- {msg}')

    print(f'\n{passed}/{len(CASES)} passed')
    if failed_cases:
        print('Failed:')
        for msg in failed_cases:
            print(f'  {msg}')
        print('FAIL')
        return 1
    print('PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
