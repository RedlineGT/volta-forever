#!/usr/bin/env python3
"""SGLang's engine, greedy, against the CPU's reference (tests/gpu/sglang.sh, stage c).

Environment:
  MODEL             path to the model (required)
  REF_FILE          JSON reference from vllm-ref.py (required): as many new tokens are asked
                    for as each of its texts has
  ATTENTION_BACKEND flashinfer, triton, torch_native (default: SGLang's own choice)
  DTYPE             float16 (default) or float32
Prints a line for each prompt, then `verdict: ` and ALL_MATCH, MATCH_BUT_NEAR_TIES n or SOME_DIFFER
(greedy_compare.py says what a near tie is).
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import greedy_compare

MODEL = os.environ['MODEL']
REF_FILE = os.environ['REF_FILE']
BACKEND = os.environ.get('ATTENTION_BACKEND') or None
DTYPE = os.environ.get('DTYPE', 'float16')


def main():
    import sglang
    ref = json.load(open(REF_FILE))
    print('sglang %s, attention backend: %s, %s' % (sglang.__version__, BACKEND or "SGLang's choice", DTYPE))
    t0 = time.time()
    options = dict(model_path=MODEL, dtype=DTYPE, mem_fraction_static=0.7, tp_size=1, log_level='info')
    if BACKEND:
        options['attention_backend'] = BACKEND
    engine = sglang.Engine(**options)
    print('engine created in %.1f s' % (time.time() - t0))
    kinds, tokens, seconds = [], 0, 0.0
    try:
        for i, r in enumerate(ref):
            t0 = time.time()
            # (No repetition penalty: the reference is plain greedy too.)
            result = engine.generate(r['prompt'], sampling_params={'temperature': 0, 'max_new_tokens': r['n_tokens'], 'repetition_penalty': 1.0})
            seconds += time.time() - t0
            text, ids = result['text'], result.get('output_ids')
            tokens += result.get('meta_info', {}).get('completion_tokens', r['n_tokens'])
            kind, detail = greedy_compare.compare(r, ids=ids, text=text)
            kinds.append(kind)
            print('  prompt %d: %s: %s' % (i, {'match': 'MATCH', 'near': 'NEAR TIE', 'differ': 'DIFFER'}[kind], detail if kind != 'match' else '%s: %r' % (detail, text[:60])))
    finally:
        try:
            engine.shutdown()
        except Exception:
            pass
    if seconds:
        print('  %d tokens in %.2f s = %.1f tok/s (one prompt at a time)' % (tokens, seconds, tokens / seconds))
    print('verdict:', greedy_compare.conclude(kinds))
    return 1 if 'differ' in kinds else 0


if __name__ == '__main__':
    sys.exit(main())
