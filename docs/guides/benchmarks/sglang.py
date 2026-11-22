#!/usr/bin/env python3
"""Measure the SGLang GPU forward phases on an otherwise idle local server."""
import argparse
import json
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
NEXTN_DECODE = ("eagle_draft", "target_verify", "eagle_draft_extend")
NEXTN_PHASES = {"extend", *NEXTN_DECODE}
CACHE_HITS = ("device_hit", "host_hit", "storage_hit")


def counters(url, model):
    values = {}
    with urlopen(url + "/metrics", timeout=10) as response:
        for line in response:
            match = METRIC.match(line.decode())
            if not match:
                continue
            name, raw_labels, value = match.groups()
            labels = dict(LABEL.findall(raw_labels))
            if labels.get("model_name") != model:
                continue
            if name == "forward_execution_seconds_total":
                key = "forward:" + labels["category"]
            elif name == "prefill_effective_tokens_total":
                key = labels["mode"]
            else:
                key = name
            values[key] = values.get(key, 0.0) + float(value)
    return values


def phase_seconds(url, model, before, input_tokens, output_tokens):
    expected = {"input": input_tokens, "prompt_tokens_total": input_tokens,
                "generation_tokens_total": output_tokens}
    deadline = time.monotonic() + 20
    previous = None
    while True:
        after = counters(url, model)
        delta = {key: value - before.get(key, 0.0) for key, value in after.items()}
        if any(value < 0 for value in delta.values()):
            raise ValueError("A server counter reset during the request.")
        if any(delta.get(key, 0) > count for key, count in expected.items()):
            raise ValueError("The counters include more tokens than this request.")
        if any(delta.get(key, 0) for key in CACHE_HITS):
            raise ValueError("SGLang reused cached prompt tokens.")
        phases = {key[8:]: value for key, value in delta.items()
                  if key.startswith("forward:") and value > 0}
        names = set(phases)
        if not (names <= ORDINARY_PHASES or names <= NEXTN_PHASES):
            raise ValueError(f"Unexpected GPU forward phases: {phases!r}")
        ready = (names in (ORDINARY_PHASES, NEXTN_PHASES) and all(key in delta for key in CACHE_HITS)
                 and all(delta.get(key) == count for key, count in expected.items()))
        if ready and previous == phases:
            if names == NEXTN_PHASES:
                return {"prefill": phases["extend"], "decode": sum(phases[k] for k in NEXTN_DECODE),
                        "speculative_forward_s": {k: phases[k] for k in NEXTN_DECODE}}
            return {"prefill": phases["extend"], "decode": phases["decode"]}
        previous = phases if ready else None
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


def measure(url, model, item):
    body = {
        "model": model, "messages": [{"role": "user", "content": item["prompt"]}],
        "temperature": 0, "max_tokens": 64, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False}, "cache_salt": uuid.uuid4().hex,
    }
    req = Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    before = counters(url, model)
    parts = []
    usage = {}
    with urlopen(req, timeout=1200) as response:
        for event in stream_events(response):
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage.update(event.get("usage") or {})
            for choice in event.get("choices", []):
                parts.append(choice["delta"].get("content") or "")
    text = "".join(parts)
    if not re.match(r"\W*" + re.escape(item["city"]) + r"\b", text, re.IGNORECASE):
        raise ValueError(f"Unexpected answer for {item['id']}: {text!r}")
    if (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0):
        raise ValueError("A request reused cached prompt tokens.")
    n_in, n_out = usage["prompt_tokens"], usage["completion_tokens"]
    if n_in < 1 or n_out < 2:
        raise ValueError(f"Insufficient tokens for {item['id']}: {usage!r}")
    seconds = phase_seconds(url, model, before, n_in, n_out)
    row = {
        "id": item["id"], "input_tokens": n_in, "output_tokens": n_out,
        "prefill_s": seconds["prefill"], "decode_s": seconds["decode"],
        "prefill_tokens_per_s": n_in / seconds["prefill"],
        "decode_tokens_per_s": (n_out - 1) / seconds["decode"], "text": text,
    }
    if "speculative_forward_s" in seconds:
        row["speculative_forward_s"] = seconds["speculative_forward_s"]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Server root URL")
    parser.add_argument("--model", default="qwen", help="Served model name")
    args = parser.parse_args()
    prompts = json.loads(Path(__file__).with_name("prompts.json").read_text())
    if [p["id"] for p in prompts] != ["warmup", "run1", "run2", "run3"]:
        raise ValueError("Expected one warm-up followed by three measured prompts.")
    results = []
    for item in prompts:
        row = measure(args.url.rstrip("/"), args.model, item)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if item["id"] != "warmup":
            results.append(row)
    median = {key: statistics.median(row[key] for row in results)
              for key in ("prefill_tokens_per_s", "decode_tokens_per_s")}
    print(json.dumps({"id": "median", **median}))


if __name__ == "__main__":
    main()
