#!/usr/bin/env python3
"""Ask SGLang's HTTP server (its own /generate) for the continuation of each prompt of a
reference, greedy, and compare it with the reference's (tests/gpu/sglang.sh, stage d).

Environment: PORT (the server listens on 127.0.0.1), REF_FILE (made by vllm-ref.py, plain greedy).
The request says: temperature 0, no repetition penalty, as many new tokens as the reference has.
Prints a line for each prompt, then `verdict: ` and ALL_MATCH, MATCH_BUT_NEAR_TIES n or SOME_DIFFER
(greedy_compare.py says what a near tie is)."""
import json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import greedy_compare

ref = json.load(open(os.environ['REF_FILE']))
url = 'http://127.0.0.1:%s/generate' % os.environ['PORT']
kinds = []
for i, r in enumerate(ref):
    body = json.dumps({'text': r['prompt'], 'sampling_params': {'temperature': 0, 'max_new_tokens': r['n_tokens'], 'repetition_penalty': 1.0}}).encode()
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'}), timeout=600) as f:
        answer = json.load(f)
    seconds = time.time() - t0
    text, ids = answer['text'], answer.get('output_ids')
    tokens = answer.get('meta_info', {}).get('completion_tokens', '?')
    kind, detail = greedy_compare.compare(r, ids=ids, text=text)
    kinds.append(kind)
    print('  prompt %d: %s (%s tokens in %.2f s): %s' % (i, {'match': 'MATCH', 'near': 'NEAR TIE', 'differ': 'DIFFER'}[kind], tokens, seconds, detail if kind != 'match' else repr(text[:60])))
    if kind != 'match' and ids is not None:
        # What the server itself gives the two candidates at that token (its own lead).
        at = next((k for k, (x, y) in enumerate(zip(ids, r['token_ids'])) if x != y), None)
        try:
            body = json.dumps({'text': r['prompt'], 'return_logprob': True, 'top_logprobs_num': 3,
                               'sampling_params': {'temperature': 0, 'max_new_tokens': at + 1, 'repetition_penalty': 1.0}}).encode()
            with urllib.request.urlopen(urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'}), timeout=600) as f:
                again = json.load(f)
            top = again['meta_info']['output_top_logprobs'][at]
            print('     the server\'s own log-probabilities at that token: %s' % ', '.join('%r %.4f' % (t[2], t[0]) for t in top))
        except Exception as e:
            print('     (the server\'s log-probabilities at that token were not read: %s)' % str(e)[:100])
print('verdict:', greedy_compare.conclude(kinds))
sys.exit(1 if 'differ' in kinds else 0)
