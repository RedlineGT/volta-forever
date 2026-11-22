#!/usr/bin/env python3
"""Measure an already running Strata or NInfer server using its native phase timers."""
import argparse
import json
import re
import statistics
import time
import urllib.request
from pathlib import Path


def api(url, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url + "/v1" + path, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1200) as response:
        return json.load(response)


def chat(engine, url, model, prompt, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if engine == "strata":
        payload.update(reasoning_effort="none", strata_checkpoint=False)
    start = time.monotonic()
    result = api(url, "/chat/completions", payload)
    wall_seconds = time.monotonic() - start
    choice = result["choices"][0]
    return {
        "answer": choice["message"]["content"],
        "finish_reason": choice["finish_reason"],
        "wall_seconds": round(wall_seconds, 4),
        "usage": result["usage"],
        "timings": result["timings"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("strata", "ninfer"))
    parser.add_argument("--url", help="Server URL, including its port")
    parser.add_argument("--prompts", type=Path, default=Path(__file__).with_name("prompts.json"))
    args = parser.parse_args()
    port = 9085 if args.engine == "strata" else 9086
    url = (args.url or f"http://127.0.0.1:{port}").rstrip("/")
    model = api(url, "/models")["data"][0]["id"]
    demo = chat(args.engine, url, model, "What is the capital of France? Answer in one word.", 12)
    if not re.search(r"\bParis\b", demo["answer"], re.IGNORECASE):
        raise SystemExit(f"Unexpected answer to the semantic check: {demo['answer']!r}")

    prompts = json.loads(args.prompts.read_text())
    if [p["id"] for p in prompts] != ["warmup", "run1", "run2", "run3"]:
        raise SystemExit("Expected the shared warm-up and three measured prompts.")
    rows = []
    for prompt in prompts:
        row = chat(args.engine, url, model, prompt["prompt"], 64)
        row["id"] = prompt["id"]
        row["first_word_matches"] = bool(re.match(r"^\W*" + re.escape(prompt["city"]) + r"\b", row["answer"], re.I))
        if not row["first_word_matches"]:
            raise SystemExit(f"Expected {prompt['city']} at the start of the reply to {row['id']}: {row['answer']!r}")
        t = row["timings"]
        if t["cache_n"] != 0 or t["prompt_n"] != row["usage"]["prompt_tokens"]:
            raise SystemExit(f"Unexpected prompt-cache reuse or inconsistent input count in {row['id']}.")
        if t["predicted_n"] != row["usage"]["completion_tokens"]:
            raise SystemExit(f"Inconsistent output count in {row['id']}.")
        if t["prompt_n"] <= 0 or t["predicted_n"] < 2 or min(t["prompt_ms"], t["predicted_ms"]) <= 0:
            raise SystemExit(f"Missing phase measurements in {row['id']}.")
        rows.append(row)

    measured = rows[1:]
    summary = {
        "median_prefill_tok_s": statistics.median(r["timings"]["prompt_per_second"] for r in measured),
        "median_decode_tok_s": statistics.median(r["timings"]["predicted_per_second"] for r in measured),
    }
    print(json.dumps({"engine": args.engine, "model": model, "semantic": demo, "runs": rows, **summary}, indent=2))


if __name__ == "__main__":
    main()
