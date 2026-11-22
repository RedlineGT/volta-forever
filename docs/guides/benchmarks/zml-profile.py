"""Report full forward-graph spans for LLMD's Triton backend on Qwen3.5-35B-A3B."""
import argparse
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Graph:
    start: int = 2**64 - 1
    end: int = 0
    kernels: int = 0
    incomplete: bool = False
    experts: Counter = field(default_factory=Counter)

    def add(self, start, end, kernel, grid):
        self.start = min(self.start, start)
        self.end = max(self.end, end)
        self.kernels += 1
        if kernel.startswith("fused_moe_kernel"):
            self.experts[kernel, grid] += 1

    @property
    def span_ms(self):
        return (self.end - self.start) / 1e6


def read_trace(trace):
    graphs = defaultdict(Graph)
    launches = {}
    markers = {}
    dropped = None
    for line in trace:
        row = line.rstrip("\n").split("\t")
        kind = row[0]
        if kind == "K":
            start, end = int(row[1]), int(row[2])
            if start == end == 0:
                # A forced flush can include a partial final decode graph.
                graphs[int(row[4])].incomplete = True
                continue
            if not 0 < start <= end:
                raise ValueError("Missing or invalid GPU kernel timestamp")
            graphs[int(row[4])].add(start, end, row[7], row[5])
        elif kind == "A" and row[5] in ("514", "515"):  # cuGraphLaunch or its per-thread-stream variant
            cid = int(row[4])
            if cid in launches:
                raise ValueError(f"Duplicate graph launch correlation ID {cid}")
            launches[cid] = int(row[1])
        elif kind == "P":
            marker = int(row[2])
            if marker in markers:
                raise ValueError(f"Duplicate request marker {marker}")
            markers[marker] = int(row[1])
        elif kind == "N":
            dropped = int(row[-1])
    if dropped != 0 or set(markers) != set(range(6)):
        raise ValueError("Expected a complete trace with six request markers and no dropped records")
    stamps = [markers[i] for i in range(6)]
    if any(a >= b for a, b in zip(stamps, stamps[1:])):
        raise ValueError("Request boundaries are out of order")
    return launches, graphs, stamps


def measure(launches, graphs, begin, end, sample):
    # This 40-layer model launches two expert projections per layer. For a single
    # 256-token prefill chunk there is one wide graph followed by single-token graphs.
    forward = []
    for cid, timestamp in sorted(launches.items(), key=lambda item: item[1]):
        if begin <= timestamp < end and cid in graphs:
            graph = graphs[cid]
            if (not graph.incomplete and len(graph.experts) == 2 and len({n for n, grid in graph.experts}) == 2
                    and all(count == 40 for count in graph.experts.values())):
                forward.append(graph)
    if len(forward) < sample["output_tokens"] - 1:
        raise ValueError(f"{sample['id']}: too few complete forward graphs")
    prefill, *decode = forward
    signature = decode[0].experts
    if prefill.experts == signature or any(g.experts != signature for g in decode):
        raise ValueError(f"{sample['id']}: could not distinguish one prefill followed by decode")
    wide = {name: math.prod(map(int, grid.split(","))) for name, grid in prefill.experts}
    narrow = {name: math.prod(map(int, grid.split(","))) for name, grid in signature}
    if wide.keys() != narrow.keys() or any(wide[name] <= narrow[name] for name in wide):
        raise ValueError(f"{sample['id']}: first graph does not contain the wider expert projections")
    largest = max(g.kernels for g in decode)
    complete = [g for g in decode if g.kernels == largest]
    if len(complete) < sample["output_tokens"] - 2:
        raise ValueError(f"{sample['id']}: truncated decode trace")
    return {"id": sample["id"], "input_tokens": sample["input_tokens"], "output_tokens": sample["output_tokens"],
            "prefill_ms": prefill.span_ms, "gpu_prefill_tokens_per_s": 1000 * sample["input_tokens"] / prefill.span_ms,
            "complete_decode_graphs": len(complete),
            "median_decode_graph_ms": statistics.median(g.span_ms for g in complete)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("responses", type=Path, help="JSON lines written by zml.py")
    args = parser.parse_args()
    opener = gzip.open if args.trace.suffix == ".gz" else open
    with opener(args.trace, "rt") as stream:
        launches, graphs, stamps = read_trace(stream)
    rows = [json.loads(line) for line in args.responses.read_text().splitlines()]
    if [r["id"] for r in rows] != ["warmup", "run1", "run2", "run3", "median"]:
        raise ValueError("Expected zml.py output from one complete run")
    rates = []
    for i, sample in enumerate(rows[1:4]):
        if (sample["input_tokens"], sample["output_tokens"]) != (227, 64):
            raise ValueError(f"{sample['id']}: unexpected token counts for the common workload")
        row = measure(launches, graphs, stamps[2*i], stamps[2*i+1], sample)
        print(json.dumps(row))
        rates.append(row["gpu_prefill_tokens_per_s"])
    print(json.dumps({"id": "median", "gpu_prefill_tokens_per_s": statistics.median(rates)}))


if __name__ == "__main__":
    main()
