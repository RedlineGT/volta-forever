#!/usr/bin/env python3
"""Measure the SGLang GPU forward phases on an otherwise idle local server.

For tensor parallelism, report the longest accumulated rank timer for each
whole phase. These counters cannot locate the slowest rank at each decode step.
Client time to first visible text and time from first to last text have separate fields.
"""
import argparse
import json
import math
import re
import statistics
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen


METRIC = re.compile(
    r"^sglang:(forward_execution_seconds_total|prefill_effective_tokens_total|"
    r"prompt_tokens_total|generation_tokens_total)\{([^}]*)\}\s+(\S+)"
)
LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')
ORDINARY_PHASES = {"extend", "decode"}
CACHE_HITS = ("device_hit", "host_hit", "storage_hit")


def parse_counters(lines, model, tp_size):
    values, rank_sets, request_series = {}, {}, set()
    for line in lines:
        match = METRIC.match(line)
        if not match:
            continue
        name, raw_labels, sample = match.groups()
        labels = dict(LABEL.findall(raw_labels))
        if labels.get("model_name") != model:
            continue
        value = float(sample)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {name} observation: {sample}")
        if name in ("forward_execution_seconds_total", "prefill_effective_tokens_total"):
            field = labels["category" if name == "forward_execution_seconds_total" else "mode"]
            rank_label = labels.get("tp_rank", "0" if tp_size == 1 else "")
            if not rank_label.isdecimal() or int(rank_label) >= tp_size:
                raise ValueError(f"Unexpected tp_rank={rank_label!r} for --tp-size {tp_size}")
            if any(labels.get(key, "0") != "0" for key in ("pp_rank", "moe_ep_rank", "dp_rank")):
                raise ValueError("This benchmark requires one pipeline, expert and data-parallel group")
            rank = int(rank_label)
            ranks = rank_sets.setdefault((name, field), set())
            if rank in ranks:
                raise ValueError(f"Duplicate rank {rank} for {name}/{field}")
            ranks.add(rank)
            # Token counts are additive. Each GPU times the same TP request.
            key = (field, rank) if name == "forward_execution_seconds_total" else field
        else:
            series = (name, tuple(sorted(labels.items())))
            if series in request_series:
                raise ValueError(f"Duplicate request counter: {series}")
            request_series.add(series)
            key = name
        values[key] = values.get(key, 0.0) + value
    for field, ranks in rank_sets.items():
        if len(ranks) != tp_size:
            raise ValueError(f"Incomplete rank series for {field}: expected {tp_size}, found {sorted(ranks)}")
    return values


def counters(url, model, tp_size):
    with urlopen(url + "/metrics", timeout=10) as response:
        return parse_counters((line.decode() for line in response), model, tp_size)


def phase_seconds(url, model, before, input_tokens, output_tokens, tp_size):
    expected = {"input": input_tokens, "prompt_tokens_total": input_tokens,
                "generation_tokens_total": output_tokens}
    deadline = time.monotonic() + 20
    previous = None
    while True:
        after = counters(url, model, tp_size)
        if before.keys() - after.keys():
            raise ValueError("A server counter disappeared during the request.")
        delta = {key: value - before.get(key, 0.0) for key, value in after.items()}
        if any(value < 0 for value in delta.values()):
            raise ValueError("A server counter reset during the request.")
        forward = {key: value for key, value in delta.items() if isinstance(key, tuple) and value > 0}
        if any(category not in ORDINARY_PHASES for category, rank in forward):
            raise ValueError(f"Unexpected speculative or non-standard forward phase: {forward!r}")
        if any(delta.get(key, 0) > count for key, count in expected.items()):
            raise ValueError("The counters include more tokens than this request.")
        if any(delta.get(key, 0) for key in CACHE_HITS):
            raise ValueError("SGLang reused cached prompt tokens.")
        ready = (len(forward) == len(ORDINARY_PHASES) * tp_size and all(key in delta for key in CACHE_HITS)
                 and all(delta.get(key) == count for key, count in expected.items()))
        if ready and previous == forward:
            return [{"tp_rank": rank, "prefill_s": forward["extend", rank], "decode_s": forward["decode", rank]}
                    for rank in range(tp_size)]
        previous = forward if ready else None
        if time.monotonic() >= deadline:
            raise ValueError(f"Missing or unsettled token counts or GPU timers: {delta!r}. "
                             "Enable --enable-metrics and SGLANG_ENABLE_METRICS_DEVICE_TIMER=1.")
        time.sleep(0.5)


def stream_events(response):
    if response.headers.get_content_type() != "text/event-stream":
        raise ValueError("Expected a streaming chat response.")
    lines = []
    for wire in response:
        line = wire.decode().rstrip("\r\n")
        if line.startswith("data:"):
            lines.append(line[5:].lstrip(" "))
        elif not line and lines:
            data = "\n".join(lines)
            if data == "[DONE]":
                return
            yield json.loads(data)
            lines.clear()
    raise ValueError("The server closed the stream before [DONE].")


def measure(url, model, item, tp_size=1):
    body = {
        "model": model, "messages": [{"role": "user", "content": item["prompt"]}],
        "temperature": 0, "max_tokens": 64, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False}, "cache_salt": uuid.uuid4().hex,
    }
    req = Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    before = counters(url, model, tp_size)
    parts = []
    usage = {}
    start = time.monotonic()
    first_text = last_text = None
    with urlopen(req, timeout=1200) as response:
        for event in stream_events(response):
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage.update(event.get("usage") or {})
            for choice in event.get("choices", []):
                content = choice["delta"].get("content")
                if content:
                    last_text = time.monotonic()
                    if first_text is None:
                        first_text = last_text
                    parts.append(content)
    text = "".join(parts)
    if not re.match(r"\W*" + re.escape(item["city"]) + r"\b", text, re.IGNORECASE):
        raise ValueError(f"Unexpected answer for {item['id']}: {text!r}")
    if (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0):
        raise ValueError("A request reused cached prompt tokens.")
    n_in, n_out = usage["prompt_tokens"], usage["completion_tokens"]
    if n_in < 1 or n_out < 2:
        raise ValueError(f"Insufficient tokens for {item['id']}: {usage!r}")
    ranks = phase_seconds(url, model, before, n_in, n_out, tp_size)
    prefill = max(row["prefill_s"] for row in ranks)
    decode = max(row["decode_s"] for row in ranks)
    result = {
        "id": item["id"], "input_tokens": n_in, "output_tokens": n_out,
        "prefill_s": prefill, "decode_s": decode,
        "prefill_tokens_per_s": n_in / prefill,
        "decode_tokens_per_s": (n_out - 1) / decode, "text": text,
    }
    if tp_size > 1:
        result["gpu_timing_method"] = "maximum accumulated GPU duration among TP ranks for each entire phase"
        result["gpu_rank_phases"] = ranks
        result["client_ttft_s"] = first_text - start
        result["client_first_to_last_text_s"] = last_text - first_text
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Server root URL")
    parser.add_argument("--model", default="qwen", help="Served model name")
    parser.add_argument("--tp-size", type=int, default=1, help="Number of tensor-parallel ranks in the server")
    args = parser.parse_args()
    if args.tp_size < 1:
        parser.error("--tp-size must be positive")
    prompts = json.loads(Path(__file__).with_name("prompts.json").read_text())
    if [p["id"] for p in prompts] != ["warmup", "run1", "run2", "run3"]:
        raise ValueError("Expected one warm-up followed by three measured prompts.")
    results = []
    for item in prompts:
        row = measure(args.url.rstrip("/"), args.model, item, args.tp_size)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if item["id"] != "warmup":
            results.append(row)
    median = {key: statistics.median(row[key] for row in results)
              for key in ("prefill_tokens_per_s", "decode_tokens_per_s")}
    print(json.dumps({"id": "median", **median}))


if __name__ == "__main__":
    main()
