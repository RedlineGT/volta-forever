#!/usr/bin/env python3
"""How tests/gpu/torch-ops.py and tests/gpu/torch-models.py run their workers, with fake workers:
no GPU and no PyTorch are needed.

    python3 tests/gpu/torch-workers.py

Each program is loaded with the path of its workers replaced by this file, which then plays a worker
that does what a scenario says: kernels that print on standard output with no end of line (as the
stub of a kernel built for a later GPU does), more on standard error than a pipe holds, a case that
does not end, a worker that dies in a case, that says it lost the GPU, that ends early or outside a
case, that does not end after its last case, a line that is not a verdict, a second verdict for a case,
a verdict for a case that it did not start or did not name. Each program must count every case
once, say what went wrong, end with the right status, and not wait for ever. The exit status is the
number of checks that failed.
"""
import importlib.util, os, signal, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
THIS = os.path.abspath(__file__)
CASES = 12          # of the fake torch-ops.py
MODELS = 6          # of the fake torch-models.py


# ---- the fake workers ---------------------------------------------------------------------------------
def noise(text):
    """What a kernel prints: on standard output, at the level of the C library, no end of line."""
    os.write(1, text.encode())


def ops_worker(first, channel, patterns):
    """A worker of torch-ops.py: cases 'case.0' to 'case.11', from `first` on, as the scenario says."""
    scenario = os.environ['FAKE_SCENARIO']
    out = os.fdopen(channel, 'w', buffering=1)

    def say(*fields):
        out.write('\t'.join(str(f) for f in fields) + '\n')
    if scenario == 'silent-start':
        time.sleep(3600)
    if scenario == 'flood':
        os.write(2, b'device-side assertion failed\n' * 10000)          # 290 kB: more than a pipe holds
    names = ['case.%d' % n for n in range(CASES)]
    selected = [n for n, name in enumerate(names) if not patterns or any(p in name for p in patterns)]
    say('cases', len(selected), ','.join(map(str, selected)))
    for number, name in enumerate(names):
        if number < first or (patterns and not any(p in name for p in patterns)):
            continue
        if scenario == 'short' and number == 9:
            return
        if scenario == 'outside' and number == 9:
            sys.exit(1)
        if scenario == 'unexpected' and number == 9:
            # A case it did not name, in place of one it named.
            say(99, 'case.99', 'start', '')
            say(99, 'case.99', 'ok', '')
            continue
        say(number, name, 'start', '')
        if scenario == 'duplicate' and number == 3:
            # A second verdict, without a start, after a DIFFERENT.
            say(number, name, 'DIFFERENT', '1 of 4 elements')
            say(number, name, 'ok', '')
            continue
        if scenario == 'mismatch' and number == 4:
            say(number, 'case.40', 'ok', '')
            continue
        if scenario == 'again' and number == 5:
            say(number, name, 'ok', '')
            say(number, name, 'start', '')
        if scenario == 'noise':
            noise('FATAL: kernel `k` is for sm80-sm121, but was built for sm75' * 3)
        if scenario == 'hang' and number == 5:
            time.sleep(3600)
        if scenario == 'died' and number == 4:
            os.kill(os.getpid(), signal.SIGKILL)
        if scenario == 'lost' and number == 2:
            say(number, name, 'CRASH', 'RuntimeError: CUDA error: an illegal memory access was encountered')
            out.flush()
            os._exit(3)
        if scenario == 'unknown' and number == 6:
            say(number, name, 'OK', '')
            continue
        if scenario == 'noise' and number in (3, 7):
            say(number, name, 'DIFFERENT', '36866 of 49152 elements')
            continue
        say(number, name, 'ok', '')
    if scenario == 'stuck-exit':
        out.close()
        time.sleep(3600)


def models_worker(name, path):
    """A worker of torch-models.py: one case of 'model.0' to 'model.5', as the scenario says."""
    scenario, number = os.environ['FAKE_SCENARIO'], int(name.split('.')[1])
    if scenario == 'flood':
        os.write(2, b'device-side assertion failed\n' * 10000)
    if scenario == 'noise':
        # (With tabs: the program took the last line with a tab on standard output as the verdict.)
        noise('ok\tfrom the CPU by 0\nFATAL: FlashAttention requires sm80\tsm90')
    if scenario == 'hang' and number == 3:
        time.sleep(3600)
    if scenario == 'none' and number == 1:
        return
    verdict = 'DIFFERENT\tfrom the CPU by 5.33e-01 of the largest value' if scenario == 'noise' and number == 2 else 'ok\tfrom the CPU by 0'
    if scenario == 'unknown' and number == 4:
        verdict = 'FATAL: FlashAttention requires sm80ok\tfrom the CPU by 0'
    with open(path, 'w') as f:
        f.write(verdict + '\n')


# ---- a program with the fake workers ----------------------------------------------------------------
def parent(program, args):
    """Run the main() of tests/gpu/<program> with this file for its workers."""
    path = os.path.join(HERE, program)
    spec = importlib.util.spec_from_file_location(program.replace('-', '_').replace('.py', ''), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__file__ = THIS
    if program == 'torch-models.py':
        module.cases = lambda: [('model.%d' % n, None, 0.0) for n in range(MODELS)]
    sys.argv = [path] + args
    return module.main()


# ---- the checks ---------------------------------------------------------------------------------------
# (program, scenario, arguments, status, lines that must be in what it prints)
CHECKS = [
    ('torch-ops.py', 'clean', [], 0, ['12 cases: 12 ok']),
    ('torch-ops.py', 'clean', ['case.1'], 0, ['3 cases: 3 ok']),
    ('torch-ops.py', 'noise', [], 1, ['12 cases: 10 ok, 2 DIFFERENT']),
    ('torch-ops.py', 'flood', [], 0, ['12 cases: 12 ok']),
    ('torch-ops.py', 'hang', [], 1, ['CRASH     case.5', 'no end within 2 s: stopped', '12 cases: 11 ok, 1 CRASH']),
    ('torch-ops.py', 'died', [], 1, ['CRASH     case.4', 'exit status -9', '12 cases: 11 ok, 1 CRASH']),
    ('torch-ops.py', 'lost', [], 1, ['CRASH     case.2', '12 cases: 11 ok, 1 CRASH']),
    ('torch-ops.py', 'unknown', [], 1, ['RUN       a worker wrote what is not a verdict', 'RUN       11 verdicts for 12 cases']),
    ('torch-ops.py', 'short', [], 1, ['RUN       9 verdicts for 12 cases', '9 cases: 9 ok']),
    ('torch-ops.py', 'outside', [], 1, ['RUN       the worker from case 0 ended with status 1 outside a case', 'RUN       9 verdicts for 12 cases']),
    ('torch-ops.py', 'stuck-exit', [], 1, ['RUN       the worker from case 0 did not end within 2 s after its last case and was stopped', '12 cases: 12 ok']),
    ('torch-ops.py', 'silent-start', [], 2, ['no worker said how many cases there are']),
    ('torch-ops.py', 'duplicate', [], 1, ['RUN       a worker gave a verdict for case 3 (case.3), which it had not started',
                                          '12 cases: 11 ok, 1 DIFFERENT']),
    ('torch-ops.py', 'unexpected', [], 1, ['RUN       a worker started case 99 (case.99), which is not one of those it named',
                                           'RUN       11 verdicts for 12 cases (none for case 9)']),
    ('torch-ops.py', 'mismatch', [], 1, ['RUN       a worker gave a verdict for case 4 (case.40), which it had not started (it had started case 4, case.4)',
                                         'RUN       11 verdicts for 12 cases (none for case 4)']),
    ('torch-ops.py', 'again', [], 1, ['RUN       a worker started case 5 (case.5) again, which has a verdict already', '12 cases: 12 ok']),
    ('torch-models.py', 'clean', [], 0, ['6 cases: 6 ok']),
    ('torch-models.py', 'noise', [], 1, ['DIFFERENT model.2', '6 cases: 5 ok, 1 DIFFERENT']),
    ('torch-models.py', 'flood', [], 0, ['6 cases: 6 ok']),
    ('torch-models.py', 'hang', [], 1, ['CRASH     model.3', 'no end within 2 s: stopped', '6 cases: 5 ok, 1 CRASH']),
    ('torch-models.py', 'none', [], 1, ['CRASH     model.1', 'the process ended with status 0', '6 cases: 5 ok, 1 CRASH']),
    ('torch-models.py', 'unknown', [], 1, ['CRASH     model.4', 'not a verdict', '6 cases: 5 ok, 1 CRASH']),
]


def main():
    args = sys.argv[1:]
    if args[:1] == ['--worker']:
        if os.environ['FAKE_PROGRAM'] == 'torch-ops.py':
            return ops_worker(int(args[1]), int(args[2]), args[3:])
        return models_worker(args[1], args[2])
    if args[:1] == ['--parent']:
        return parent(args[1], args[2:])
    failed = 0
    with tempfile.TemporaryDirectory(prefix='vf-torch-workers-') as out:
        for program, scenario, arguments, status, expected in CHECKS:
            env = dict(os.environ, FAKE_PROGRAM=program, FAKE_SCENARIO=scenario, VF_CASE_WAIT='2', VF_OUT=out)
            # (In a session of its own, so that what is left of it can be stopped if it waits for ever.)
            p = subprocess.Popen([sys.executable, THIS, '--parent', program] + arguments, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            try:
                printed, _ = p.communicate(timeout=60)
                wrong = [] if p.returncode == status else ['status %d, not %d' % (p.returncode, status)]
                wrong += ['nothing like %r' % e for e in expected if e not in printed]
            except subprocess.TimeoutExpired:
                printed, wrong = '', ['still waiting after 60 s']
            finally:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
            name = '%s %s%s' % (program, scenario, ''.join(' ' + a for a in arguments))
            print('%-4s %s%s' % ('FAIL' if wrong else 'ok', name, ': ' + '; '.join(wrong) if wrong else ''))
            if wrong:
                failed += 1
                print('     ' + '\n     '.join(printed.strip().split('\n')[-12:]))
    print('%d of %d checks pass' % (len(CHECKS) - failed, len(CHECKS)))
    return failed


if __name__ == '__main__':
    sys.exit(main())
