#!/usr/bin/env python3
"""Measure prompt tokens per server TTFT and post-first-token decode throughput on an idle LLMD server."""
import argparse
import json
import math
import re
import socket
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

METRICS = {
    "prefill_tokens_total", "output_tokens_total", "decode_steps_total", "decode_seconds_total",
    "gen_ai_server_time_to_first_token_count", "gen_ai_server_time_to_first_token_sum",
}


def counters(url, model):
    values = {}
    with urlopen(url + "/metrics", timeout=15) as response:
        for line in response:
            name, _, rest = line.decode().partition("{")
            if name in METRICS:
                labels, _, number = rest.partition("} ")
                if labels == "model_name=" + json.dumps(model):
                    value = float(number.strip())
                    if name in values or not math.isfinite(value):
                        raise ValueError(f"Invalid server counter {name}")
                    values[name] = value
    return values


def measure(url, model, item):
    body = {
        "model": model, "messages": [{"role": "user", "content": item["prompt"]}],
        "temperature": 0, "reasoning_effort": "none", "max_tokens": 64, "stream": True, "stream_options": {"include_usage": True},
    }
    request = Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    before = counters(url, model)
    parts = []
    usage = {}
    with urlopen(request, timeout=300) as response:
        if response.headers.get_content_type() != "text/event-stream":
            raise ValueError("Expected LLMD to stream the answer")
        lines = []
        for wire in response:
            line = wire.decode().rstrip("\r\n")
            if line.startswith("data:"):
                lines.append(line[5:].lstrip(" "))
            elif not line and lines:
                data = "\n".join(lines)
                lines.clear()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(event["error"])
                usage.update(event.get("usage") or {})
                for choice in event.get("choices", []):
                    parts.append(choice["delta"].get("content") or "")
        else:
            raise ValueError("LLMD closed the stream before the final event")
    text = "".join(parts)
    if not re.match(r"\s*" + re.escape(item["city"]) + r"\b", text, re.IGNORECASE):
        raise ValueError(f"Unexpected answer for {item['id']}: {text!r}")
    n_in = usage["prompt_tokens"]
    n_out = usage["completion_tokens"]
    if n_in <= 0 or n_out < 2:
        raise ValueError(f"Insufficient tokens: {usage!r}")
    deadline = time.monotonic() + 10
    while True:
        after = counters(url, model)
        delta = {k: v - before.get(k, 0) for k, v in after.items()}
        observations = delta.get("gen_ai_server_time_to_first_token_count", 0)
        if observations > 1 or any(v < 0 for v in delta.values()):
            raise ValueError("LLMD restarted or served another request during the benchmark")
        if observations == 1 and delta.get("output_tokens_total") == n_out and METRICS <= delta.keys():
            break
        if time.monotonic() >= deadline:
            raise ValueError(f"Incomplete LLMD counters for {item['id']}: {delta!r}")
        time.sleep(0.05)
    if delta["prefill_tokens_total"] != n_in or delta["decode_steps_total"] not in (n_out - 1, n_out):
        raise ValueError(f"Expected an uncached prompt and one decode step per token: {delta!r}")
    ttft = delta["gen_ai_server_time_to_first_token_sum"]
    decode = delta["decode_seconds_total"]
    if min(ttft, decode) <= 0:
        raise ValueError(f"Expected positive LLMD durations: {delta!r}")
    return {"id": item["id"], "input_tokens": n_in, "output_tokens": n_out,
            "server_ttft_s": ttft, "decode_steps": int(delta["decode_steps_total"]),
            "decode_s": decode, "prompt_tokens_per_ttft_s": n_in / ttft,
            "decode_tokens_per_s": (n_out - 1) / decode,
            "decode_steps_per_s": delta["decode_steps_total"] / decode, "reply": text}


def mark_profile(directory, number):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as control:
        control.settimeout(120)
        control.connect(str(directory / "control"))
        marker = str(number).encode("ascii")
        control.sendall(marker)
        if control.recv(1) != marker:
            raise RuntimeError(f"LLMD failed to record profiling marker {number}")


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--url", default="http://127.0.0.1:8000")
    args.add_argument("--model", default="qwen")
    args.add_argument("--profile-dir", type=Path, help="directory of the optional CUPTI capture")
    options = args.parse_args()
    prompts = json.loads(Path(__file__).with_name("prompts.json").read_text())
    if [item["id"] for item in prompts] != ["warmup", "run1", "run2", "run3"]:
        raise ValueError("Expected a warm-up and three measured prompts")
    measured = []
    for index, item in enumerate(prompts):
        if options.profile_dir and index:
            mark_profile(options.profile_dir, 2 * (index - 1))
        row = measure(options.url.rstrip("/"), options.model, item)
        if options.profile_dir and index:
            mark_profile(options.profile_dir, 2 * index - 1)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if item["id"] != "warmup":
            measured.append(row)
    median = {k: statistics.median(row[k] for row in measured)
              for k in ["prompt_tokens_per_ttft_s", "decode_tokens_per_s", "decode_steps_per_s"]}
    print(json.dumps({"id": "median", **median}))


if __name__ == "__main__":
    main()
