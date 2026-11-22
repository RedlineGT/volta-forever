#!/usr/bin/env python3
"""vLLM engine test: start the engine and generate text.

Environment:
  MODEL       path to the model (required)
  DTYPE       float16 or float32 (default float16)
  MAX_TOKENS  maximum new tokens (default 50)
  TEMPERATURE 0 for greedy (default), >0 for sampling
  TOP_K       top-k for sampling (default: not set)
  TOP_P       top-p for sampling (default: not set)
  SEED        random seed for sampling (default 42)
  REF_FILE    JSON reference from vllm-ref.py (optional; compare greedy tokens: prints ALL_MATCH,
              MATCH_BUT_NEAR_TIES n or SOME_DIFFER, as greedy_compare.py defines them)
With sampling and TOP_K: every token sampled must be among the TOP_K most likely, and within the
TOP_P of their probability at TEMPERATURE, by the log-probabilities of the model's own logits that
the engine gives with each token (`raw_logprobs`): what the kernels of the sampling did is checked,
not what the model computed (that is the greedy stage's). Prints ALL_WITHIN or SOME_OUTSIDE.
Prints PASS or FAIL with the reason.
"""
import sys, os, json, math, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import greedy_compare

os.environ.setdefault('DO_NOT_TRACK', '1')
os.environ.setdefault('VLLM_NO_USAGE_STATS', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

MODEL = os.environ['MODEL']
DTYPE = os.environ.get('DTYPE', 'float16')
MAX_TOKENS = int(os.environ.get('MAX_TOKENS', '50'))
TEMPERATURE = float(os.environ.get('TEMPERATURE', '0'))
TOP_K = int(os.environ['TOP_K']) if 'TOP_K' in os.environ else -1
TOP_P = float(os.environ['TOP_P']) if 'TOP_P' in os.environ else 1.0
SEED = int(os.environ.get('SEED', '42'))
REF_FILE = os.environ.get('REF_FILE', '')

# (How much more than TOP_P the tokens before a sampled one may make, by this computation in
# float64 of what the engine computes in float32 and cuts by its own method.)
SLACK = 0.01

PROMPTS = [
    "The capital of France is",
    "What is 2 + 2? The answer is",
    "Explain briefly what a GPU is:",
]

def outside(logprobs, chosen):
    """None when token `chosen` is among the TOP_K most likely by `logprobs` (the engine's
    {token: Logprob} at one place, of the logits before any processing), and the more likely of
    those make less than TOP_P (and SLACK) of their probability at TEMPERATURE; else why not."""
    ranked = sorted(logprobs.items(), key=lambda kv: -kv[1].logprob)[:TOP_K]
    tokens = [token for token, _ in ranked]
    if chosen not in tokens:
        return 'token %d is not among the %d most likely (its rank: %s)' % (chosen, TOP_K, logprobs[chosen].rank if chosen in logprobs else '?')
    weights = [math.exp((lp.logprob - ranked[0][1].logprob) / TEMPERATURE) for _, lp in ranked]
    before = sum(weights[:tokens.index(chosen)]) / sum(weights)
    if before >= TOP_P + SLACK:
        return 'token %d comes after tokens that make %.4f of the probability, more than %g' % (chosen, before, TOP_P)
    return None


def main():
    from vllm import LLM, SamplingParams

    sp_kwargs = dict(max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
    checked = TEMPERATURE > 0 and TOP_K > 0
    if TEMPERATURE > 0:
        sp_kwargs['seed'] = SEED
        if TOP_K > 0:
            sp_kwargs['top_k'] = TOP_K
        if TOP_P < 1.0:
            sp_kwargs['top_p'] = TOP_P
        if checked:
            sp_kwargs['logprobs'] = TOP_K
    sampling_params = SamplingParams(**sp_kwargs)

    try:
        t0 = time.time()
        llm = LLM(model=MODEL, dtype=DTYPE, trust_remote_code=True,
                   gpu_memory_utilization=0.8, max_model_len=512, enforce_eager=True,
                   **({'max_logprobs': TOP_K, 'logprobs_mode': 'raw_logprobs'} if checked else {}))
        t_engine = time.time() - t0
        print(f'Engine created in {t_engine:.1f}s')

        t0 = time.time()
        outputs = llm.generate(PROMPTS, sampling_params)
        t_gen = time.time() - t0

        total_tokens = 0
        kinds, wrong = [], []
        ref = None
        if REF_FILE and os.path.exists(REF_FILE):
            ref = json.load(open(REF_FILE))

        for i, o in enumerate(outputs):
            text = o.outputs[0].text
            token_ids = list(o.outputs[0].token_ids)
            n = len(token_ids)
            total_tokens += n
            print(f'  prompt {i}: {n} tokens: {text[:80]!r}')
            if ref and i < len(ref):
                kind, detail = greedy_compare.compare(ref[i], ids=token_ids)
                kinds.append(kind)
                print('    %s: %s' % ({'match': 'MATCH', 'near': 'NEAR TIE', 'differ': 'DIFFER'}[kind], detail))
            if checked:
                places = o.outputs[0].logprobs or []
                why = None if token_ids else 'no token'
                for k, token in enumerate(token_ids):
                    w = outside(places[k], token) if k < len(places) else 'no log-probabilities'
                    if w:
                        why = 'place %d: %s' % (k, w)
                        break
                wrong.append(why)
                print('    %s' % ('OUTSIDE: ' + why if why else 'WITHIN: %d tokens, each among the %d most likely and within %g of their probability' % (n, TOP_K, TOP_P)))

        if t_gen > 0:
            print(f'  {total_tokens} tokens in {t_gen:.2f}s = {total_tokens/t_gen:.1f} tok/s')

        if ref:
            print(greedy_compare.conclude(kinds))
        if checked:
            print('SOME_OUTSIDE' if any(wrong) else 'ALL_WITHIN')

        print('PASS')
    except Exception as e:
        print(f'FAIL: {e}')
        traceback.print_exc()

if __name__ == '__main__':
    main()
