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
  COMPILE     1: vLLM's compilation on (`enforce_eager` off, and TORCH_COMPILE_DISABLE=0 unless
              it is set); by default it is off, as the stages of vllm.sh run the engine
  IDS_OUT     a file to write the token ids of every prompt to, as JSON (optional)
  IDS_REF     such a file of another run (optional): prints, for each prompt, how many tokens
              are the same before the first that differs, and `same tokens: ALL` or `SOME_DIFFER`
  REF_FILE    JSON reference from vllm-ref.py (optional; compare greedy tokens: prints
              `verdict: ALL_MATCH`, `verdict: MATCH_BUT_NEAR_TIES n` or `verdict: SOME_DIFFER`, as
              greedy_compare.py defines them)
There must be one result for each prompt; with fewer (or none), the verdict is SOME_DIFFER or
SOME_OUTSIDE.
With sampling and TOP_K: every token sampled must be among the TOP_K most likely, and within the
TOP_P of their probability at TEMPERATURE, by the log-probabilities of the model's own logits that
the engine gives with each token (`raw_logprobs`): what the kernels of the sampling did is checked,
not what the model computed (that is the greedy stage's). Prints `verdict: ALL_WITHIN` or
`verdict: SOME_OUTSIDE`. A log-probability that is not a number, or that says more than a
probability of 1, or log-probabilities that make more than 1 together, are outside too.
Prints PASS or FAIL with the reason; the exit status is 1 after FAIL.
"""
import sys, os, json, math, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import greedy_compare

os.environ.setdefault('DO_NOT_TRACK', '1')
os.environ.setdefault('VLLM_NO_USAGE_STATS', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
COMPILE = os.environ.get('COMPILE') == '1'
os.environ.setdefault('TORCH_COMPILE_DISABLE', '0' if COMPILE else '1')

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
    those make less than TOP_P (and SLACK) of their probability at TEMPERATURE; else why not.
    The log-probabilities are checked first: each a number, none above a probability of 1 (and
    SLACK), all of them together not above 1 (and SLACK), the chosen one's not of a probability
    of 0. (A NaN compares false with everything: until 2026-11-23 a token whose log-probability
    was NaN passed.)"""
    for token, lp in logprobs.items():
        if math.isnan(lp.logprob) or lp.logprob > math.log1p(SLACK):
            return 'token %d has the log-probability %r, which is not one' % (token, lp.logprob)
    total = math.fsum(math.exp(lp.logprob) for lp in logprobs.values())
    if total > 1 + SLACK:
        return 'the log-probabilities given make %.4f of the probability together, more than 1' % total
    ranked = sorted(logprobs.items(), key=lambda kv: -kv[1].logprob)[:TOP_K]
    tokens = [token for token, _ in ranked]
    if chosen not in tokens:
        return 'token %d is not among the %d most likely (its rank: %s)' % (chosen, TOP_K, logprobs[chosen].rank if chosen in logprobs else '?')
    if logprobs[chosen].logprob == -math.inf:
        return 'token %d was sampled with a probability of 0' % chosen
    weights = [math.exp((lp.logprob - ranked[0][1].logprob) / TEMPERATURE) for _, lp in ranked]
    before = sum(weights[:tokens.index(chosen)]) / sum(weights)
    if not before < TOP_P + SLACK:
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
                   gpu_memory_utilization=0.8, max_model_len=512, enforce_eager=not COMPILE,
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
            elif ref:
                kinds.append('differ')
                print('    DIFFER: the reference has no text for this prompt')
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

        ids = [list(o.outputs[0].token_ids) for o in outputs]
        if os.environ.get('IDS_OUT'):
            json.dump(ids, open(os.environ['IDS_OUT'], 'w'))
        if os.environ.get('IDS_REF'):
            other = json.load(open(os.environ['IDS_REF']))
            same = True
            for i, (mine, theirs) in enumerate(zip(ids, other)):
                first = next((k for k, (a, b) in enumerate(zip(mine, theirs)) if a != b), min(len(mine), len(theirs)))
                whole = first == len(mine) == len(theirs)
                same &= whole
                print(f'  prompt {i}: {first} of {len(mine)} tokens the same as the other run\'s' + ('' if whole else ' before the first that differs'))
            print('same tokens:', 'ALL' if same and len(ids) == len(other) else 'SOME_DIFFER')

        # One result for each prompt: a verdict over fewer, or over none, is not one.
        if len(outputs) != len(PROMPTS):
            print(f'  {len(outputs)} results for {len(PROMPTS)} prompts')
            kinds.append('differ')
            wrong.append('%d results for %d prompts' % (len(outputs), len(PROMPTS)))

        if t_gen > 0:
            print(f'  {total_tokens} tokens in {t_gen:.2f}s = {total_tokens/t_gen:.1f} tok/s')

        if ref:
            print('verdict:', greedy_compare.conclude(kinds))
        if checked:
            print('verdict:', 'SOME_OUTSIDE' if any(wrong) else 'ALL_WITHIN')

        print('PASS')
        return 0
    except Exception as e:
        print(f'FAIL: {e}')
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
