#!/usr/bin/env python3
"""Collectives of NCCL through PyTorch, one process for each GPU, checked against the CPU.

    python3 tests/gpu/nccl-collectives.py [--bench] [--world N]

Run by tests/gpu/nccl.sh, which says which NCCL and which layer. Each rank makes its inputs from
its rank alone, so that every rank computes on the CPU what each collective must give; integers,
and floats that hold integers small enough to be exact in every partial sum, so that the answer is
exact whatever order NCCL adds in. Each case prints one line from rank 0: PASS NAME, or FAIL NAME
and the first values that differ on any rank. --bench then times an all-reduce of 64 MiB of
float32 and one of 8 bytes (medians of CUDA events over each rank, the slowest rank's median
said). The last line is "ALL CHECKS PASSED" or "CHECKS FAILED: N"; the status is 0, 1, or 2 when
fewer than N GPUs are visible.
"""
import argparse
import os
import socket
import sys
import time


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def cases(torch, dist, rank, world):
    """(name, function) for each case; a function returns (got, expected) CPU tensors."""
    dev = torch.device("cuda", rank)

    def ints(n, dtype, r, scale=1, offset=0):
        # Distinct values for each rank and position, small enough for the sums to be exact.
        base = torch.arange(n, dtype=torch.int64) % 1009
        return ((base * (r + 1) * scale + r * 7 + offset) % 2048).to(dtype)

    out = []
    for n in (1, 257, 65536, 1 << 22):
        for dtype in (torch.int32, torch.int64, torch.float32, torch.float16, torch.bfloat16):
            if dtype in (torch.float16, torch.bfloat16) and n > 65536:
                continue

            def all_reduce_sum(n=n, dtype=dtype):
                # bf16 holds integers up to 256 exactly, f16 up to 2048: keep the sums there.
                cap = {torch.bfloat16: 64, torch.float16: 512}.get(dtype)
                make = (lambda r: (ints(n, torch.int64, r) % cap).to(dtype)) if cap else (lambda r: ints(n, dtype, r))
                t = make(rank).to(dev)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                expected = sum(make(r).to(torch.int64) for r in range(world))
                return t.cpu().to(torch.int64), expected

            out.append((f"all_reduce SUM {str(dtype)[6:]} x{n}", all_reduce_sum))
        for op_name, op, fold in (("MAX", dist.ReduceOp.MAX, torch.maximum), ("MIN", dist.ReduceOp.MIN, torch.minimum)):

            def all_reduce_op(n=n, op=op, fold=fold):
                t = ints(n, torch.int32, rank, scale=3, offset=11).to(dev)
                dist.all_reduce(t, op=op)
                expected = ints(n, torch.int32, 0, scale=3, offset=11)
                for r in range(1, world):
                    expected = fold(expected, ints(n, torch.int32, r, scale=3, offset=11))
                return t.cpu(), expected

            out.append((f"all_reduce {op_name} int32 x{n}", all_reduce_op))

        def broadcast(n=n):
            root = world - 1
            t = (ints(n, torch.int32, rank) if rank == root else torch.zeros(n, dtype=torch.int32)).to(dev)
            dist.broadcast(t, src=root)
            return t.cpu(), ints(n, torch.int32, root)

        def all_gather(n=n):
            t = ints(n, torch.int32, rank).to(dev)
            gathered = torch.empty(n * world, dtype=torch.int32, device=dev)
            dist.all_gather_into_tensor(gathered, t)
            return gathered.cpu(), torch.cat([ints(n, torch.int32, r) for r in range(world)])

        def reduce_scatter(n=n):
            t = ints(n * world, torch.int32, rank).to(dev)
            part = torch.empty(n, dtype=torch.int32, device=dev)
            dist.reduce_scatter_tensor(part, t, op=dist.ReduceOp.SUM)
            whole = sum(ints(n * world, torch.int64, r) for r in range(world))
            return part.cpu().to(torch.int64), whole[rank * n : (rank + 1) * n]

        def reduce(n=n):
            t = ints(n, torch.int64, rank).to(dev)
            dist.reduce(t, dst=0, op=dist.ReduceOp.SUM)
            if rank != 0:
                return torch.zeros(1), torch.zeros(1)
            return t.cpu(), sum(ints(n, torch.int64, r) for r in range(world))

        def send_recv(n=n):
            # Each rank sends to the next and receives from the one before.
            t = ints(n, torch.int32, rank, offset=5).to(dev)
            got = torch.empty(n, dtype=torch.int32, device=dev)
            ops = [
                dist.P2POp(dist.isend, t, (rank + 1) % world),
                dist.P2POp(dist.irecv, got, (rank - 1) % world),
            ]
            for request in dist.batch_isend_irecv(ops):
                request.wait()
            return got.cpu(), ints(n, torch.int32, (rank - 1) % world, offset=5)

        def all_to_all(n=n):
            t = ints(n * world, torch.int32, rank, offset=3).to(dev)
            got = torch.empty(n * world, dtype=torch.int32, device=dev)
            dist.all_to_all_single(got, t)
            expected = torch.cat([ints(n * world, torch.int32, r, offset=3)[rank * n : (rank + 1) * n] for r in range(world)])
            return got.cpu(), expected

        out += [
            (f"broadcast int32 x{n}", broadcast),
            (f"all_gather int32 x{n}", all_gather),
            (f"reduce_scatter SUM int32 x{n}", reduce_scatter),
            (f"reduce SUM int64 x{n}", reduce),
            (f"send/recv int32 x{n}", send_recv),
            (f"all_to_all int32 x{n}", all_to_all),
        ]
    return out


def loaded_nccl():
    """The NCCL that this process loaded: its file and the version that it says (ncclGetVersion).
    PyTorch's torch.cuda.nccl.version() is the version of the headers it was built with."""
    import ctypes

    with open("/proc/self/maps") as maps:
        paths = sorted({line.split()[-1] for line in maps if "/libnccl" in line and ".so" in line})
    for path in paths:
        try:
            library = ctypes.CDLL(path)
            code = ctypes.c_int()
            if library.ncclGetVersion(ctypes.byref(code)) == 0:
                v = code.value
                return path, f"{v // 10000}.{v % 10000 // 100}.{v % 100}"
        except OSError:
            continue
    return ", ".join(paths) or "none found", "unknown"


def median(values):
    values = sorted(values)
    return values[len(values) // 2]


def bench(torch, dist, rank, world):
    dev = torch.device("cuda", rank)
    lines = []
    for label, count, turns in (("64 MiB float32", 1 << 24, 20), ("8 bytes float32", 2, 200)):
        t = torch.ones(count, dtype=torch.float32, device=dev)
        for _ in range(5):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        times = []
        for _ in range(turns):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            dist.all_reduce(t)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) / 1000.0)
        mine = torch.tensor([median(times)], dtype=torch.float64, device=dev)
        dist.all_reduce(mine, op=dist.ReduceOp.MAX)
        seconds = mine.item()
        size = count * 4
        bus = size * 2 * (world - 1) / world / seconds / 1e9
        lines.append(f"BENCH all_reduce {label}: median {seconds * 1e6:.1f} us, bus bandwidth {bus:.2f} GB/s")
    return lines


def rank_main(rank, world, port, args, results):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    lines = []
    failed = 0
    try:
        dist.init_process_group(
            "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world, device_id=torch.device("cuda", rank)
        )
        if rank == 0:
            path, version = loaded_nccl()
            lines.append(f"NCCL {version} ({path}), PyTorch {torch.__version__}, {world} ranks")
        for name, case in cases(torch, dist, rank, world):
            started = time.monotonic()
            got, expected = case()
            torch.cuda.synchronize()
            wrong = int((got != expected).sum()) if got.shape == expected.shape else -1
            flag = torch.tensor([1 if wrong else 0], dtype=torch.int32, device=torch.device("cuda", rank))
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if wrong:
                at = (got != expected).nonzero()[:3].flatten().tolist() if wrong > 0 else []
                lines.append(f"rank {rank}: {name}: {wrong} values differ (first at {at})")
            if rank == 0:
                ok = flag.item() == 0
                failed += not ok
                lines.append(f"{'PASS' if ok else 'FAIL'}  {name} ({time.monotonic() - started:.2f} s)")
        if args.bench:
            bench_lines = bench(torch, dist, rank, world)
            if rank == 0:
                lines += bench_lines
        dist.destroy_process_group()
    except Exception as e:  # what stops a rank is a result too
        lines.append(f"rank {rank}: ERROR {type(e).__name__}: {e}")
        failed += 1
    results.put((rank, lines, failed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=2)
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    import torch
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < args.world:
        print(f"SKIP: {torch.cuda.device_count()} GPUs visible, {args.world} needed")
        return 2
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    port = free_port()
    procs = [ctx.Process(target=rank_main, args=(r, args.world, port, args, results)) for r in range(args.world)]
    for p in procs:
        p.start()
    gathered = {}
    for _ in procs:
        try:
            rank, lines, failed = results.get(timeout=float(os.environ.get("NCCL_TEST_TIMEOUT", "500")))
        except Exception:
            break
        gathered[rank] = (lines, failed)
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.kill()
    failed = 0
    for rank in range(args.world):
        lines, rank_failed = gathered.get(rank, ([f"rank {rank}: no result"], 1))
        for line in lines:
            print(line, flush=True)
        failed += rank_failed
    failed += sum(1 for p in procs if p.exitcode not in (0, None))
    print("ALL CHECKS PASSED" if failed == 0 else f"CHECKS FAILED: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
