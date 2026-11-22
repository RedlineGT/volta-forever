#!/usr/bin/env python3
"""Ask vLLM's HTTP server (the OpenAI API) for a completion of each prompt of a reference, greedy,
and compare the text with the reference's (tests/gpu/vllm.sh, stage d).

Environment: PORT (the server listens on 127.0.0.1), REF_FILE (made by vllm-ref.py; as many tokens
are asked for as each of its texts has), MODEL_NAME (default `model`).
The request names no repetition penalty: the server takes the one of the model's configuration,
which is what the reference must have been made with (PENALTY=model).
Prints a line for each prompt, then `verdict: ` and ALL_MATCH, MATCH_BUT_NEAR_TIES n or SOME_DIFFER
(greedy_compare.py says what a near tie is)."""
import json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import greedy_compare

ref = json.load(open(os.environ['REF_FILE']))
url = 'http://127.0.0.1:%s/v1/completions' % os.environ['PORT']
kinds = []
for i, r in enumerate(ref):
    body = json.dumps({'model': os.environ.get('MODEL_NAME', 'model'), 'prompt': r['prompt'],
                       'max_tokens': r['n_tokens'], 'temperature': 0}).encode()
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'}), timeout=300) as f:
        answer = json.load(f)
    seconds = time.time() - t0
    text, tokens = answer['choices'][0]['text'], answer['usage']['completion_tokens']
    kind, detail = greedy_compare.compare(r, text=text)
    kinds.append(kind)
    print('  prompt %d: %s (%d tokens in %.2f s): %s' % (i, {'match': 'MATCH', 'near': 'NEAR TIE', 'differ': 'DIFFER'}[kind], tokens, seconds, detail if kind != 'match' else repr(text[:60])))
print('verdict:', greedy_compare.conclude(kinds))
sys.exit(1 if 'differ' in kinds else 0)
