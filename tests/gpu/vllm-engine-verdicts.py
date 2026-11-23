#!/usr/bin/env python3
"""The verdicts of tests/gpu/vllm-engine.py, with a stand-in for vLLM: no GPU, no vLLM.

The stand-in's engine gives, for the prompts it is asked, what the scenario says: one result for
each, none, results for two of the three, or a sampled token outside the top k. Each scenario is
run greedy (against a reference with the same tokens, or with texts for two prompts only) and
sampling (TOP_K), and the verdict line the program prints must be the one expected: a verdict over
fewer results than prompts, or over none, does not pass. (Until 2026-11-23 the program checked the
results that it got, and none gave `verdict: ALL_WITHIN`.)

Run with python3 tests/gpu/vllm-engine-verdicts.py.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PROGRAM = Path(__file__).resolve().parent / "vllm-engine.py"
TOKENS = [11, 12, 13]

STANDIN = '''
import os
class SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
class Logprob:
    def __init__(self, logprob, rank):
        self.logprob, self.rank = logprob, rank
class Completion:
    def __init__(self, token_ids, logprobs):
        self.token_ids, self.text, self.logprobs = token_ids, 'text', logprobs
class Result:
    def __init__(self, completion):
        self.outputs = [completion]
class LLM:
    def __init__(self, **kwargs):
        pass
    def generate(self, prompts, params):
        scenario = os.environ['VF_TEST_SCENARIO']
        tokens = %r
        # The token sampled at each place is the most likely, unless the scenario says otherwise.
        chosen = [99 if scenario == 'outside' else t for t in tokens]
        places = [{t: Logprob(-0.1, 1), t + 100: Logprob(-2.0, 2)} for t in tokens]
        results = [Result(Completion(chosen, places)) for _ in prompts]
        return {'one': results, 'outside': results, 'none': [], 'fewer': results[:2]}[scenario]
''' % (TOKENS,)

# (scenario, sampling, reference prompts, the verdict expected)
CASES = [
    ('one', False, 3, 'ALL_MATCH'),
    ('none', False, 3, 'SOME_DIFFER'),
    ('fewer', False, 3, 'SOME_DIFFER'),
    ('one', False, 2, 'SOME_DIFFER'),
    ('one', True, 0, 'ALL_WITHIN'),
    ('outside', True, 0, 'SOME_OUTSIDE'),
    ('none', True, 0, 'SOME_OUTSIDE'),
    ('fewer', True, 0, 'SOME_OUTSIDE'),
]


def main():
    wrong = 0
    with tempfile.TemporaryDirectory() as tmp:
        site = Path(tmp) / "site"
        (site / "vllm").mkdir(parents=True)
        (site / "vllm/__init__.py").write_text(STANDIN)
        for scenario, sampling, prompts, want in CASES:
            env = {k: v for k, v in os.environ.items() if k not in ("REF_FILE", "TOP_K", "TOP_P", "TEMPERATURE")}
            env.update(PYTHONPATH=str(site), MODEL="stand-in", VF_TEST_SCENARIO=scenario)
            if sampling:
                env.update(TEMPERATURE="0.8", TOP_K="5")
            else:
                reference = Path(tmp) / f"reference-{prompts}.json"
                reference.write_text(json.dumps([{"text": "text", "token_ids": TOKENS}] * prompts))
                env["REF_FILE"] = str(reference)
            output = subprocess.run([sys.executable, str(PROGRAM)], env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, timeout=60).stdout
            said = [line[len("verdict: "):] for line in output.splitlines() if line.startswith("verdict: ")]
            got = said[-1] if said else None
            case = f"{scenario} {'sampling' if sampling else f'greedy, a reference of {prompts}'}"
            if got == want:
                print(f"ok   {case}: {got}")
            else:
                wrong += 1
                print(f"WRONG {case}: {got}, want {want}\n{output[-2000:]}")
    print(f"{len(CASES) - wrong} of {len(CASES)} checks pass")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
