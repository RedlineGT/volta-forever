#!/usr/bin/env python3
"""The verdicts of tests/gpu/vllm-engine.py, with a stand-in for vLLM: no GPU, no vLLM; and the
verdict of a stage of tests/gpu/vllm.sh and tests/gpu/sglang.sh (tests/gpu/verdict.sh).

The stand-in's engine gives, for the prompts it is asked, what the scenario says: one result for
each, none, results for two of the three, a sampled token outside the top k, a log-probability that
is NaN or above 0, an error. Each scenario is run greedy (against a reference with the same tokens,
or with texts for two prompts only) and sampling (TOP_K), and the verdict line the program prints
must be the one expected: a verdict over fewer results than prompts, or over none, does not pass,
nor one over a log-probability that is not one. (Until 2026-11-23 the program checked the results
that it got, and none gave `verdict: ALL_WITHIN`; and a token whose log-probability was NaN was
within.) Then verdict.sh is given what a stage's program printed and its status, as vllm.sh gives
them: a verdict passes only when the program then ends by itself with status 0 within its time
(until 2026-11-23, a program that printed `verdict: ALL_WITHIN` and hung as it ended passed).

Run with python3 tests/gpu/vllm-engine-verdicts.py.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PROGRAM = Path(__file__).resolve().parent / "vllm-engine.py"
VERDICT = Path(__file__).resolve().parent / "verdict.sh"
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
        if scenario == 'error':
            raise RuntimeError('the engine stopped')
        if scenario == 'hang':
            # The verdict is printed, then the program does not end (an engine's teardown).
            import atexit, time
            atexit.register(time.sleep, 3600)
        # The token sampled at each place is the most likely, unless the scenario says otherwise;
        # its log-probability -0.1 (of the two given, 0.95 of the probability), unless the
        # scenario says otherwise.
        chosen = [99 if scenario == 'outside' else t for t in tokens]
        mine = {'nan': float('nan'), 'above': 0.5}.get(scenario, -0.1)
        places = [{t: Logprob(mine, 1), t + 100: Logprob(-3.0, 2)} for t in tokens]
        results = [Result(Completion(chosen, places)) for _ in prompts]
        return {'none': [], 'fewer': results[:2]}.get(scenario, results)
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
    ('nan', True, 0, 'SOME_OUTSIDE'),
    ('above', True, 0, 'SOME_OUTSIDE'),
    ('error', True, 0, None),
]

# What verdict.sh says of a stage: (what the program does, the verdict it waits for, the start of
# the line expected, and the function of verdict.sh that judges: `verdict` when not said). The
# program is run as vllm.sh runs one, here `timeout -k 1 3`, and its output and status are given to
# that function. `verdict_or_limit` (stage e of vllm.sh) says LIMIT, and leaves `failed` as it was,
# for a program that ended with an error after the layer's line of a kernel that only the code for
# the capability told has; through `verdict`, the same is a FAIL.
NOT_GIVEN = ('echo "[volta-forever] kernel k is not in the code that the layer gives for its image (x), its machine '
             'code for sm_75 translated: it is only in the image\'s machine code for sm_80"')
FAKE = {
    'passes': 'echo "verdict: ALL_WITHIN"',
    'has near ties': 'echo "verdict: MATCH_BUT_NEAR_TIES 2"',
    'hangs after its verdict': 'echo "verdict: ALL_WITHIN"; sleep 60',
    'hangs after its verdict, deaf to SIGTERM': 'trap "" TERM; echo "verdict: ALL_WITHIN"; sleep 60',
    'fails after its verdict': 'echo "verdict: ALL_WITHIN"; exit 3',
    'is refused a kernel': 'echo "[volta-forever] kernel k cannot run on this GPU and is not given to the program: x"; exit 1',
    'gives another verdict': 'echo "verdict: SOME_OUTSIDE"',
    'stops after a kernel of the code told': NOT_GIVEN + '; echo "FAIL: CUDA error: named symbol not found"; exit 1',
    'hangs after a kernel of the code told': NOT_GIVEN + '; sleep 60',
    'is told of a kernel of the code told, and passes': NOT_GIVEN + '; echo "verdict: ALL_MATCH"',
    'stops after a kernel of the code told and a refusal': NOT_GIVEN + '; echo "[volta-forever] kernel r cannot run on '
                                                           'this GPU and is not given to the program: x"; exit 1',
}
STAGES = [
    ('passes', 'ALL_WITHIN', 'PASS stage-x'),
    ('has near ties', 'ALL_MATCH', 'PASS stage-x (with 2 near tie(s)'),
    ('hangs after its verdict', 'ALL_WITHIN', 'FAIL stage-x: ran out of time ($VF_ENGINE_WAIT is 3 s), after the verdict ALL_WITHIN'),
    ('hangs after its verdict, deaf to SIGTERM', 'ALL_WITHIN', 'FAIL stage-x: ran out of time ($VF_ENGINE_WAIT is 3 s), after the verdict ALL_WITHIN'),
    ('fails after its verdict', 'ALL_WITHIN', 'FAIL stage-x: the program ended with status 3, after the verdict ALL_WITHIN'),
    ('is refused a kernel', 'ALL_WITHIN', 'REFUSED stage-x: [volta-forever] kernel k cannot run'),
    ('gives another verdict', 'ALL_WITHIN', 'FAIL stage-x'),
    ('stops after a kernel of the code told', 'ALL_MATCH', 'FAIL stage-x: the program ended with status 1'),
    ('stops after a kernel of the code told', 'ALL_MATCH', 'LIMIT stage-x: [volta-forever] kernel k is not in the code',
     'verdict_or_limit'),
    ('hangs after a kernel of the code told', 'ALL_MATCH', 'FAIL stage-x: ran out of time', 'verdict_or_limit'),
    ('is told of a kernel of the code told, and passes', 'ALL_MATCH', 'PASS stage-x', 'verdict_or_limit'),
    ('stops after a kernel of the code told and a refusal', 'ALL_MATCH', 'REFUSED stage-x: [volta-forever] kernel r',
     'verdict_or_limit'),
    ('passes', 'ALL_WITHIN', 'PASS stage-x', 'verdict_or_limit'),
    ('fails after its verdict', 'ALL_WITHIN', 'FAIL stage-x: the program ended with status 3', 'verdict_or_limit'),
]


def stage(command, word, judge='verdict'):
    """What verdict.sh prints, then `failed=N`, for `command` (a shell command, or an argv) run as
    vllm.sh runs a stage's program, judged by the function `judge` of verdict.sh."""
    script = ('source "$1"; engine_wait=3; failed=0; judge=$2; shift 3; out=$(timeout -k 1 "$engine_wait" "$@" 2>&1); '
              'status=$?; "$judge" x "$out" "$0" "$status"; echo "failed=$failed"')
    argv = ["sh", "-c", command] if isinstance(command, str) else command
    return subprocess.run(["bash", "-c", script, word, str(VERDICT), judge, "--", *argv], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=60).stdout


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
            run = subprocess.run([sys.executable, str(PROGRAM)], env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, timeout=60)
            output = run.stdout
            said = [line[len("verdict: "):] for line in output.splitlines() if line.startswith("verdict: ")]
            got = said[-1] if said else None
            case = f"{scenario} {'sampling' if sampling else f'greedy, a reference of {prompts}'}"
            # (No verdict and status 1 after FAIL; status 0 after a verdict.)
            if got == want and (run.returncode == 0) == (want is not None):
                print(f"ok   {case}: {got}, status {run.returncode}")
            else:
                wrong += 1
                print(f"WRONG {case}: {got}, status {run.returncode}; want {want}\n{output[-2000:]}")
        # The stand-in's program prints its verdict and does not end: through verdict.sh, no pass.
        env = {k: v for k, v in os.environ.items() if k not in ("REF_FILE", "TOP_K", "TOP_P", "TEMPERATURE")}
        env.update(PYTHONPATH=str(site), MODEL="stand-in", VF_TEST_SCENARIO="hang", TEMPERATURE="0.8", TOP_K="5")
        checks = [(name, word, want, FAKE[name], (judge or ['verdict'])[0]) for name, word, want, *judge in STAGES] + [
            ('is vllm-engine.py and hangs after its verdict', 'ALL_WITHIN',
             'FAIL stage-x: ran out of time ($VF_ENGINE_WAIT is 3 s), after the verdict ALL_WITHIN',
             ["env", "-i", *[f"{k}={v}" for k, v in env.items()], sys.executable, str(PROGRAM)], 'verdict')]
        for name, word, want, command, judge in checks:
            said = stage(command, word, judge)
            first = said.splitlines()[0] if said else ''
            failed = 'failed=0' not in said
            if first.startswith(want) and failed == (not want.startswith(('PASS', 'LIMIT'))):
                print(f"ok   the stage verdict ({judge}) of a program that {name}: {first}")
            else:
                wrong += 1
                print(f"WRONG the stage verdict ({judge}) of a program that {name}: {said!r}, want {want!r}")
    total = len(CASES) + len(checks)
    print(f"{total - wrong} of {total} checks pass")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
