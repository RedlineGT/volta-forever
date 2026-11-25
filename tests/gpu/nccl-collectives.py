#!/usr/bin/env python3
"""Collectives of NCCL through PyTorch, one process for each GPU, checked against the CPU.

    python3 tests/gpu/nccl-collectives.py [--bench] [--protocols] [--world N]

Run by tests/gpu/nccl.sh, which says which NCCL and which layer. Each rank makes its inputs from
its rank alone, so that every rank computes on the CPU what each collective must give, and every
value is compared exactly, in its type: nothing has a tolerance. Integers; and floats (float64,
float32, float16, bfloat16) that are multiples of a power of two small enough for every sum of
them over up to eight ranks to be a value of the type, so that a sum is exact whatever order NCCL
adds in (the expected values are made in float64, which holds them, and given the type: exact);
the maximum and the minimum of floats are exact anyway. Each case prints one line from rank 0:
PASS NAME, or FAIL NAME and the first values that differ on any rank.

NCCL chooses its algorithm and protocol by itself for the first pass. --protocols adds a pass for
each algorithm and protocol pinned (NCCL_ALGO, NCCL_PROTO in the environment of a new group of the
ranks, which NCCL reads when it makes its communicator): Ring with Simple, LL and LL128 (every
collective), Tree with the three (the all-reduces: the others have no tree), each named in its
cases' lines; a pair that NCCL does not take on this machine is said, not failed (NCCL's own lines
say why, with NCCL_DEBUG=WARN). --bench then times an all-reduce of 64 MiB of float32 and one of
8 bytes (medians of CUDA events over each rank, the slowest rank's median said). The last line is
"ALL CHECKS PASSED" or "CHECKS FAILED: N"; the status is 0, 1, or 2 when fewer than N GPUs are
visible.
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


def cases(torch, dist, rank, world, group=None, sizes=(1, 257, 65536, 1 << 22), all_reduce_only=False):
    """(name, function) for each case; a function returns (got, expected) CPU tensors of one type."""
    dev = torch.device("cuda", rank)

    def ints(n, dtype, r, scale=1, offset=0):
        # Distinct values for each rank and position, small enough for the sums to be exact.
        base = torch.arange(n, dtype=torch.int64) % 1009
        return ((base * (r + 1) * scale + r * 7 + offset) % 2048).to(dtype)

    # For each float type, the step of its values and their bound: (k % (2 * bound + 1) - bound) * step,
    # so that every sum of eight of them is a multiple of the step below 8 * bound * step, a value of
    # the type (float16: 11 bits, bfloat16: 8, float32: 24, float64: 53).
    steps = {torch.float64: (2.0**-20, 1000), torch.float32: (2.0**-10, 1000),
             torch.float16: (2.0**-3, 64), torch.bfloat16: (2.0**-2, 8)}

    def floats(n, dtype, r, scale=1):
        step, bound = steps[dtype]
        k = (torch.arange(n, dtype=torch.int64) * (r + 1) * scale + r * 13) % (2 * bound + 1) - bound
        return k.to(torch.float64) * step

    out = []
    for n in sizes:
        for dtype in (torch.int32, torch.int64, torch.float64, torch.float32, torch.float16, torch.bfloat16):
            if dtype in (torch.float16, torch.bfloat16) and n > 65536:
                continue
            is_float = dtype.is_floating_point

            def all_reduce_sum(n=n, dtype=dtype, is_float=is_float):
                make = (lambda r: floats(n, dtype, r)) if is_float else (lambda r: ints(n, torch.int64, r))
                t = make(rank).to(dtype).to(dev)
                dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
                # (In float64 for floats, which holds every sum: then of the type, exactly.)
                expected = sum(make(r) for r in range(world)).to(dtype)
                return t.cpu(), expected

            out.append((f"all_reduce SUM {str(dtype)[6:]} x{n}", all_reduce_sum))
            for op_name, op, fold in (("MAX", dist.ReduceOp.MAX, torch.maximum), ("MIN", dist.ReduceOp.MIN, torch.minimum)):
                if dtype == torch.int64:
                    continue

                def all_reduce_op(n=n, op=op, fold=fold, dtype=dtype, is_float=is_float):
                    def make(r):
                        return floats(n, dtype, r, scale=3) if is_float else ints(n, torch.int64, r, scale=3, offset=11)

                    t = make(rank).to(dtype).to(dev)
                    dist.all_reduce(t, op=op, group=group)
                    expected = make(0)
                    for r in range(1, world):
                        expected = fold(expected, make(r))
                    return t.cpu(), expected.to(dtype)

                out.append((f"all_reduce {op_name} {str(dtype)[6:]} x{n}", all_reduce_op))
        if all_reduce_only:
            continue

        def broadcast(n=n):
            root = world - 1
            t = (ints(n, torch.int32, rank) if rank == root else torch.zeros(n, dtype=torch.int32)).to(dev)
            dist.broadcast(t, src=root, group=group)
            return t.cpu(), ints(n, torch.int32, root)

        def all_gather(n=n):
            t = ints(n, torch.int32, rank).to(dev)
            gathered = torch.empty(n * world, dtype=torch.int32, device=dev)
            dist.all_gather_into_tensor(gathered, t, group=group)
            return gathered.cpu(), torch.cat([ints(n, torch.int32, r) for r in range(world)])

        def reduce_scatter(n=n):
            t = floats(n * world, torch.float32, rank).to(torch.float32).to(dev)
            part = torch.empty(n, dtype=torch.float32, device=dev)
            dist.reduce_scatter_tensor(part, t, op=dist.ReduceOp.SUM, group=group)
            whole = sum(floats(n * world, torch.float32, r) for r in range(world)).to(torch.float32)
            return part.cpu(), whole[rank * n : (rank + 1) * n]

        def reduce(n=n):
            t = ints(n, torch.int64, rank).to(dev)
            dist.reduce(t, dst=0, op=dist.ReduceOp.SUM, group=group)
            if rank != 0:
                return torch.zeros(1), torch.zeros(1)
            return t.cpu(), sum(ints(n, torch.int64, r) for r in range(world))

        def send_recv(n=n):
            # Each rank sends to the next and receives from the one before.
            t = ints(n, torch.int32, rank, offset=5).to(dev)
            got = torch.empty(n, dtype=torch.int32, device=dev)
            ops = [
                dist.P2POp(dist.isend, t, (rank + 1) % world, group=group),
                dist.P2POp(dist.irecv, got, (rank - 1) % world, group=group),
            ]
            for request in dist.batch_isend_irecv(ops):
                request.wait()
            return got.cpu(), ints(n, torch.int32, (rank - 1) % world, offset=5)

        def all_to_all(n=n):
            t = ints(n * world, torch.int32, rank, offset=3).to(dev)
            got = torch.empty(n * world, dtype=torch.int32, device=dev)
            dist.all_to_all_single(got, t, group=group)
            expected = torch.cat([ints(n * world, torch.int32, r, offset=3)[rank * n : (rank + 1) * n] for r in range(world)])
            return got.cpu(), expected

        out += [
            (f"broadcast int32 x{n}", broadcast),
            (f"all_gather int32 x{n}", all_gather),
            (f"reduce_scatter SUM float32 x{n}", reduce_scatter),
            (f"reduce SUM int64 x{n}", reduce),
            (f"send/recv int32 x{n}", send_recv),
            (f"all_to_all int32 x{n}", all_to_all),
        ]
    return out


# The algorithms and protocols pinned by --protocols: (NCCL_ALGO, NCCL_PROTO, the all-reduces only).
PINNED = [(algo, proto, algo == "Tree") for algo in ("Ring", "Tree") for proto in ("Simple", "LL", "LL128")]


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
        passes = [("", None, cases(torch, dist, rank, world))]
        if args.protocols:
            for algo, proto, all_reduce_only in PINNED:
                passes.append((f"{algo}/{proto}", (algo, proto), None))
        for label, pinned, made in passes:
            group = None
            if pinned is not None:
                # NCCL reads these when it makes the group's communicator (eagerly, here).
                os.environ["NCCL_ALGO"], os.environ["NCCL_PROTO"] = pinned
                try:
                    group = dist.new_group(list(range(world)), device_id=torch.device("cuda", rank))
                    made = cases(torch, dist, rank, world, group, sizes=(257, 65536, 1 << 22),
                                 all_reduce_only=pinned[0] == "Tree")
                    # A first collective: NCCL says here when it has no algorithm for the pair.
                    probe = torch.ones(4, device=torch.device("cuda", rank))
                    dist.all_reduce(probe, group=group)
                    torch.cuda.synchronize()
                except Exception as e:
                    if rank == 0:
                        lines.append(f"NOT TAKEN  NCCL_ALGO={pinned[0]} NCCL_PROTO={pinned[1]}: {type(e).__name__}: {e}")
                    continue
                finally:
                    del os.environ["NCCL_ALGO"], os.environ["NCCL_PROTO"]
            for name, case in made:
                name = f"{name} [{label}]" if label else name
                started = time.monotonic()
                got, expected = case()
                torch.cuda.synchronize()
                same = got.dtype == expected.dtype and got.shape == expected.shape
                wrong = int((got != expected).sum()) if same else -1
                flag = torch.tensor([1 if wrong else 0], dtype=torch.int32, device=torch.device("cuda", rank))
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                if wrong:
                    at = (got != expected).nonzero()[:3].flatten().tolist() if wrong > 0 else []
                    lines.append(f"rank {rank}: {name}: {wrong} values differ (first at {at})")
                if rank == 0:
                    ok = flag.item() == 0
                    failed += not ok
                    lines.append(f"{'PASS' if ok else 'FAIL'}  {name} ({time.monotonic() - started:.2f} s)")
            if group is not None:
                dist.destroy_process_group(group)
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
    parser.add_argument("--protocols", action="store_true")
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
