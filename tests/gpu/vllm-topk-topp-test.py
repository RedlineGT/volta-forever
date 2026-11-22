#!/usr/bin/env python3
"""Test the _topk_topp_kernel against a CPU reference.

The kernel applies combined top-k and top-p filtering to logits in-place:
  - top-k: keep only the k highest logits, mask the rest to -inf
  - top-p: among the top-k survivors, keep only the smallest set whose
    softmax probabilities sum to at least p, mask the rest to -inf
  - When both are active, top-k is applied first, then top-p within the survivors.

This script compares the Triton kernel (through the layer on a V100) with a
PyTorch CPU reference for several batch sizes, vocabulary sizes, k and p values.

Environment:
  VF_PROXY must be set (the proxy's lib directory).
  PYTHONPATH must include vllm and torch sites.
Prints PASS or FAIL per case, and a summary; the exit status is 1 when a case fails.
"""
import os, sys, torch, math

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

def run_triton_topk_topp(logits_gpu, k_gpu, p_gpu, mask_value, vocab_size):
    """Run the actual vLLM Triton kernel on GPU."""
    from vllm.v1.sample.ops.topk_topp_triton import _topk_topp_kernel
    from vllm.platforms import current_platform
    from vllm.utils.platform_utils import num_compute_units

    batch_size = logits_gpu.shape[0]

    topk_enabled = k_gpu is not None
    topp_enabled = p_gpu is not None
    k_ptr = k_gpu if k_gpu is not None else logits_gpu
    p_ptr = p_gpu if p_gpu is not None else logits_gpu

    block_size = 8192
    block_size_trunc = 4096
    num_warps = 8
    num_sm = num_compute_units()
    num_programs = min(num_sm, batch_size)

    buffer = torch.empty((num_programs, vocab_size), dtype=torch.float32, device='cuda')

    # Build the lookup tables
    from vllm.v1.sample.ops.topk_topp_triton import (
        _PERCENTILE_TO_STD_TABLE, _NORMAL_CDF_TO_SIGMA_TABLE
    )
    pct_table = torch.tensor(_PERCENTILE_TO_STD_TABLE, dtype=torch.float32, device='cuda')
    cdf_table = torch.tensor(_NORMAL_CDF_TO_SIGMA_TABLE, dtype=torch.float32, device='cuda')

    split_covers_ponly = (
        current_platform.is_cuda_alike()
        and topp_enabled
        and batch_size <= 64
    )

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
    torch.cuda.synchronize()
    return logits_gpu


def compare_results(gpu_logits, cpu_logits, label):
    """Compare GPU and CPU results. Returns (pass, message)."""
    # Both should have the same set of non-masked positions
    gpu_alive = gpu_logits > float('-inf')
    cpu_alive = cpu_logits > float('-inf')

    if not torch.equal(gpu_alive, cpu_alive):
        # Count differences
        gpu_count = gpu_alive.sum().item()
        cpu_count = cpu_alive.sum().item()
        diff_positions = (gpu_alive != cpu_alive).sum().item()
        return False, f'{label}: alive set differs: gpu={gpu_count} cpu={cpu_count} diff_positions={diff_positions}'

    # Where both are alive, compare values
    both_alive = gpu_alive & cpu_alive
    if both_alive.sum().item() == 0:
        return True, f'{label}: both all-masked (trivially equal)'

    gpu_vals = gpu_logits[both_alive]
    cpu_vals = cpu_logits[both_alive]

    if torch.equal(gpu_vals, cpu_vals):
        return True, f'{label}: EXACT match ({both_alive.sum().item()} alive)'

    max_diff = (gpu_vals - cpu_vals).abs().max().item()
    return False, f'{label}: values differ, max_diff={max_diff:.6e} ({both_alive.sum().item()} alive)'


def run_test_case(batch_size, vocab_size, k_val, p_val, seed=42):
    """Run one test case. Returns (pass, message)."""
    torch.manual_seed(seed)
    logits_cpu = torch.randn(batch_size, vocab_size, dtype=torch.float32)

    # Prepare k and p
    k_cpu = None
    p_cpu = None
    if k_val is not None:
        if isinstance(k_val, int):
            k_cpu = torch.full((batch_size,), k_val, dtype=torch.int32)
        else:
            k_cpu = torch.tensor(k_val, dtype=torch.int32)
    if p_val is not None:
        if isinstance(p_val, (int, float)):
            p_cpu = torch.full((batch_size,), p_val, dtype=torch.float32)
        else:
            p_cpu = torch.tensor(p_val, dtype=torch.float32)

    # CPU reference: account for the kernel's SPLIT_COVERS_PONLY behavior.
    # When TOPK_ENABLED, TOPP_ENABLED, batch_size <= 64 (on CUDA), and k >= vocab_size
    # for a given row, the kernel skips that row (the split pipeline handles it).
    # Our test does not run the split pipeline, so the GPU leaves all logits alive.
    topk_enabled = k_cpu is not None
    topp_enabled = p_cpu is not None
    split_covers_ponly = topk_enabled and topp_enabled and batch_size <= 64

    logits_ref = logits_cpu.clone()
    if split_covers_ponly and k_cpu is not None:
        # For rows where k >= vocab_size, the kernel does nothing (split covers them)
        # For rows where k < vocab_size, the kernel applies top-k + top-p as normal
        for row in range(batch_size):
            if k_cpu[row].item() >= vocab_size:
                # kernel does nothing for this row: logits stay as-is
                pass
            else:
                row_logits = logits_ref[row:row+1]
                topk_topp_cpu(row_logits,
                              k=k_cpu[row:row+1].clone() if k_cpu is not None else None,
                              p=p_cpu[row:row+1].clone() if p_cpu is not None else None)
    else:
        topk_topp_cpu(logits_ref, k=k_cpu.clone() if k_cpu is not None else None,
                      p=p_cpu.clone() if p_cpu is not None else None)

    # GPU
    logits_gpu = logits_cpu.clone().cuda()
    k_gpu = k_cpu.cuda() if k_cpu is not None else None
    p_gpu = p_cpu.cuda() if p_cpu is not None else None

    try:
        run_triton_topk_topp(logits_gpu, k_gpu, p_gpu, float('-inf'), vocab_size)
    except Exception as e:
        return False, f'batch={batch_size} vocab={vocab_size} k={k_val} p={p_val}: GPU error: {e}'

    gpu_result = logits_gpu.cpu()
    label = f'batch={batch_size} vocab={vocab_size} k={k_val} p={p_val}'
    return compare_results(gpu_result, logits_ref, label)


def main():
    mask_value = float('-inf')
    total = 0
    passed = 0
    failed_cases = []

    # Test cases: (batch_size, vocab_size, k, p, description)
    cases = [
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

        # edge: k disabled (k >= vocab_size), p active
        # NOTE: when TOPK_ENABLED=True and SPLIT_COVERS_PONLY=True (batch <= 64 on CUDA),
        # the kernel skips standalone top-p for rows where k >= VOCAB_SIZE, because those
        # rows are handled by the split pipeline (_topp_split_* kernels). So the kernel
        # intentionally leaves all logits alive. We test this as the kernel's correct behavior.
        (1, 32000, 32000, 0.9, 'k=vocab (split covers, no-op expected)'),
        (1, 32000, 100000, 0.9, 'k>vocab (split covers, no-op expected)'),

        # ties: logits with many equal values
    ]

    # Add a ties case
    torch.manual_seed(99)
    # We'll handle this specially below

    for batch_size, vocab_size, k_val, p_val, desc in cases:
        ok, msg = run_test_case(batch_size, vocab_size, k_val, p_val)
        total += 1
        if ok:
            passed += 1
            print(f'  PASS: {desc} -- {msg}')
        else:
            failed_cases.append(msg)
            print(f'  FAIL: {desc} -- {msg}')

    # Ties case: many duplicates
    torch.manual_seed(99)
    batch_size, vocab_size = 1, 1000
    logits_cpu = torch.zeros(batch_size, vocab_size, dtype=torch.float32)
    # Only 10 distinct values
    logits_cpu[0, :100] = 1.0
    logits_cpu[0, 100:200] = 0.5
    logits_cpu[0, 200:300] = 0.0
    k_cpu = torch.tensor([50], dtype=torch.int32)

    logits_ref = logits_cpu.clone()
    topk_topp_cpu(logits_ref, k=k_cpu.clone())

    logits_gpu = logits_cpu.clone().cuda()
    k_gpu = k_cpu.cuda()
    try:
        run_triton_topk_topp(logits_gpu, k_gpu, None, float('-inf'), vocab_size)
        torch.cuda.synchronize()
        gpu_result = logits_gpu.cpu()
        # For ties, we compare the number of survivors rather than exact positions
        gpu_alive = (gpu_result > float('-inf')).sum().item()
        cpu_alive = (logits_ref > float('-inf')).sum().item()
        total += 1
        if gpu_alive == cpu_alive:
            passed += 1
            print(f'  PASS: ties (k=50 from 100 equal top values) -- alive: gpu={gpu_alive} cpu={cpu_alive}')
        else:
            msg = f'ties: alive count differs: gpu={gpu_alive} cpu={cpu_alive}'
            failed_cases.append(msg)
            print(f'  FAIL: {msg}')
    except Exception as e:
        total += 1
        msg = f'ties: GPU error: {e}'
        failed_cases.append(msg)
        print(f'  FAIL: {msg}')

    print(f'\n{passed}/{total} passed')
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
