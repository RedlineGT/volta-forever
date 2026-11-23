#!/usr/bin/env python3
"""Whole models of PyTorch on the GPU against the same models on the CPU.

    torch-models.py [-v] [PATTERN...]     (run by tests/gpu/torch-models.sh, which sets the paths)
    torch-models.py --list
    torch-models.py --comparison          (what decides ok, checked on tensors of the CPU)

tests/gpu/torch-ops.py takes operations one by one. This runs small models end to end, as a program
would: an image classifier (convolutions, normalisation, pooling), a transformer encoder and a causal
language model (attention, with heads of 64 and of 128), recurrent networks, an upsampling network;
each in f32, under autocast (f16 on the GPU, where the stock wheel takes its tensor-core kernels),
and through a few steps of training (backward passes, an optimizer, a gradient scaler). The weights
and the inputs are made on the CPU from a fixed seed, so two installations of the same PyTorch work on
the same numbers.

Each case runs in a process of its own (a fault on the GPU ends the process, not the run) and ends as

    ok        the GPU's result is the CPU's within the case's tolerance, in the first run (in which
              kernels are compiled and caches filled) and in a second
    DIFFERENT it is not
    refused   a kernel it needs is not given to the program (its function is not translated)
    error     PyTorch raised something else on the GPU
    CRASH     the process ended in the case

With -v the line of every case says how far the GPU is from the CPU (the largest difference as a
fraction of the largest value) and how long the GPU took. A refused case names the kernel and why
its function is not translated, as the proxy says it. The exit status is 1 if a case is DIFFERENT
or a CRASH.

A case writes its verdict to a file of its own: kernels print on standard output, and a message
that runs into a verdict would hide it. A process that ends without a verdict, or with a line that
is not one, is a CRASH; so is one that has not ended after $VF_CASE_WAIT seconds (default 600): it
is stopped.
"""
import os, re, signal, subprocess, sys, tempfile


def cases():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    out = []

    def case(name, build, tolerance):
        out.append((name, build, tolerance))

    def seeded(n):
        torch.manual_seed(20261118 + n)

    # ---- models ---------------------------------------------------------------------------------------
    class Block(nn.Module):
        def __init__(self, cin, cout, stride):
            super().__init__()
            self.a = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
            self.na = nn.BatchNorm2d(cout)
            self.b = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
            self.nb = nn.BatchNorm2d(cout)
            self.skip = None if stride == 1 and cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

        def forward(self, x):
            y = F.relu(self.na(self.a(x)))
            y = self.nb(self.b(y))
            return F.relu(y + (x if self.skip is None else self.skip(x)))

    class Classifier(nn.Module):
        """A residual network of 8 convolutions: maximum pooling after the stem, `pool` at the end."""
        def __init__(self, pool):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(3, 32, 7, 2, 3, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(3, 2, 1))
            self.body = nn.Sequential(Block(32, 32, 1), Block(32, 64, 2), Block(64, 128, 2))
            self.pool = pool
            self.head = nn.Linear(128, 10)

        def forward(self, x):
            return self.head(self.pool(self.body(self.stem(x))).flatten(1))

    class Encoder(nn.Module):
        def __init__(self, width, heads, layers=2, vocabulary=1000):
            super().__init__()
            self.embed = nn.Embedding(vocabulary, width)
            layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.head = nn.Linear(width, vocabulary)

        def forward(self, tokens):
            return self.head(self.body(self.embed(tokens)))

    class CausalBlock(nn.Module):
        def __init__(self, width, heads):
            super().__init__()
            self.heads = heads
            self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
            self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
            self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

        def forward(self, x):
            b, t, c = x.shape
            q, k, v = (z.view(b, t, self.heads, c // self.heads).transpose(1, 2) for z in self.qkv(self.n1(x)).chunk(3, -1))
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.proj(y.transpose(1, 2).reshape(b, t, c))
            return x + self.mlp(self.n2(x))

    class Causal(nn.Module):
        def __init__(self, width, heads, layers=2, vocabulary=1000, context=256):
            super().__init__()
            self.tokens, self.places = nn.Embedding(vocabulary, width), nn.Embedding(context, width)
            self.body = nn.Sequential(*[CausalBlock(width, heads) for _ in range(layers)])
            self.norm, self.head = nn.LayerNorm(width), nn.Linear(width, vocabulary, bias=False)

        def forward(self, tokens):
            x = self.tokens(tokens) + self.places(torch.arange(tokens.shape[1], device=tokens.device))
            return self.head(self.norm(self.body(x)))

    class Recurrent(nn.Module):
        def __init__(self, kind):
            super().__init__()
            self.embed = nn.Embedding(500, 64)
            self.body = kind(64, 96, num_layers=2, batch_first=True, bidirectional=True)
            self.head = nn.Linear(192, 20)

        def forward(self, tokens):
            return self.head(self.body(self.embed(tokens))[0])

    class Upsampler(nn.Module):
        """Down by a convolution with a stride, up again by interpolation, with a skip."""
        def __init__(self, mode):
            super().__init__()
            self.mode = mode
            self.down = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.GroupNorm(4, 16), nn.SiLU())
            self.mid = nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1), nn.GroupNorm(4, 16), nn.SiLU())
            self.out = nn.Conv2d(16 + 3, 3, 3, 1, 1)

        def forward(self, x):
            y = self.mid(self.down(x))
            kw = {} if self.mode == 'nearest' else {'align_corners': False}
            y = F.interpolate(y, size=x.shape[-2:], mode=self.mode, **kw)
            return self.out(torch.cat([y, x], 1))

    # ---- what a case does with a model ---------------------------------------------------------------
    def inference(make, inputs, half=False):
        """The model's output on the device, in f32 or under autocast."""
        def run(device):
            model = make().eval().to(device)
            x = inputs().to(device)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=half and device.type == 'cuda'):
                return model(x).float()
        return run

    def training(make, inputs, targets, steps=4, half=False, optimizer='adamw'):
        """The losses of a few steps and the output after them."""
        def run(device):
            model = make().train().to(device)
            x, y = inputs().to(device), targets().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3) if optimizer == 'adamw' else torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
            amp = half and device.type == 'cuda'
            scaler = torch.amp.GradScaler('cuda', enabled=amp)
            losses = []
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                    out = model(x)
                    loss = F.cross_entropy(out.float().reshape(-1, out.shape[-1]), y.reshape(-1))
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                losses.append(loss.detach().float().reshape(1))
            model.eval()
            with torch.no_grad():
                return torch.cat(losses), model(x).float()
        return run

    def built(n, fn):
        """`fn` with the generator at the same place whatever ran before: models and inputs made on the CPU."""
        def make():
            seeded(n)
            return fn()
        return make

    images = built(1, lambda: torch.randn(8, 3, 64, 64))
    labels10 = built(2, lambda: torch.randint(0, 10, (8,)))
    tokens = built(3, lambda: torch.randint(0, 1000, (4, 128)))
    next_tokens = built(4, lambda: torch.randint(0, 1000, (4, 128)))
    words = built(5, lambda: torch.randint(0, 500, (6, 40)))
    tags = built(6, lambda: torch.randint(0, 20, (6, 40)))

    for name, pool in (('adaptive', lambda: nn.AdaptiveAvgPool2d(1)), ('avgpool', lambda: nn.AvgPool2d(4))):
        make = built(10, lambda pool=pool: Classifier(pool()))
        case('classifier.%s.f32' % name, inference(make, images), 2e-3)
        case('classifier.%s.f16' % name, inference(make, images, half=True), 3e-2)
    make = built(10, lambda: Classifier(nn.AdaptiveAvgPool2d(1)))
    case('classifier.train.f32', training(make, images, labels10, optimizer='sgd'), 2e-2)
    case('classifier.train.f16', training(make, images, labels10, half=True, optimizer='sgd'), 1e-1)

    for width, heads in ((256, 4), (512, 4), (192, 6)):
        size = '%d' % (width // heads)
        make = built(20, lambda width=width, heads=heads: Encoder(width, heads))
        case('encoder.h%s.f32' % size, inference(make, tokens), 2e-3)
        case('encoder.h%s.f16' % size, inference(make, tokens, half=True), 3e-2)
        make = built(21, lambda width=width, heads=heads: Causal(width, heads))
        case('causal.h%s.f32' % size, inference(make, tokens), 2e-3)
        case('causal.h%s.f16' % size, inference(make, tokens, half=True), 3e-2)
        case('causal.h%s.train.f32' % size, training(make, tokens, next_tokens), 2e-2)
        case('causal.h%s.train.f16' % size, training(make, tokens, next_tokens, half=True), 1e-1)

    for name, kind in (('lstm', nn.LSTM), ('gru', nn.GRU)):
        make = built(30, lambda kind=kind: Recurrent(kind))
        case('%s.f32' % name, inference(make, words), 2e-3)
        case('%s.train.f32' % name, training(make, words, tags), 2e-2)

    for mode in ('nearest', 'bilinear'):
        make = built(40, lambda mode=mode: Upsampler(mode))
        case('upsampler.%s.f32' % mode, inference(make, images), 2e-3)
        case('upsampler.%s.f16' % mode, inference(make, images, half=True), 3e-2)
    return out


def worker(name, path):
    """One case: 'VERDICT<tab>DETAIL' to the file `path`."""
    import faulthandler
    import time
    import torch

    def say(line):
        with open(path, 'w') as f:
            f.write(line + '\n')
    # (Where it was, on standard error, when it is stopped for taking too long.)
    faulthandler.register(signal.SIGUSR1)
    build, tolerance = next((b, t) for n, b, t in cases() if n == name)
    reference = build(torch.device('cpu'))
    try:
        device = torch.device('cuda')
        start = time.perf_counter()
        cold = build(device)
        torch.cuda.synchronize()
        first = time.perf_counter() - start
        start = time.perf_counter()
        warm = build(device)
        torch.cuda.synchronize()
        again = time.perf_counter() - start
    except Exception as e:                                   # noqa: BLE001 (whatever PyTorch raises is the result)
        text = str(e).split('\n')[0][:160]
        refused = 'PTX JIT compilation failed' in str(e) or 'invalid device function' in str(e)
        say('%s\t%s' % ('refused' if refused else 'error', '' if refused else '%s: %s' % (type(e).__name__, text)))
        return
    ok, detail = verdict([cold, warm], reference, tolerance)
    say('%s\t%s; %.0f ms, then %.0f ms' % ('ok' if ok else 'DIFFERENT', detail, 1000 * first, 1000 * again))


def verdict(runs, reference, tolerance):
    """(ok, what to say) of the results of the runs on the GPU (the first, in which kernels are
    compiled and caches filled, then a second) against the CPU's: each run, as many tensors as the
    CPU's, each of the shape and type of the CPU's (compared before any arithmetic, which would
    broadcast one shape to the other), within `tolerance` of the largest value of the CPU's, none
    NaN or infinite."""
    import torch

    def tensors(result):
        return result if isinstance(result, tuple) else (result,)
    reference = tensors(reference)
    found = []
    for result in map(tensors, runs):
        if len(result) != len(reference):
            found.append((None, 0, '%d tensors where the CPU has %d' % (len(result), len(reference))))
            continue
        other = next(((tuple(got.shape), got.dtype, tuple(want.shape), want.dtype) for got, want in zip(result, reference)
                      if tuple(got.shape) != tuple(want.shape) or got.dtype != want.dtype), None)
        if other:
            found.append((None, 0, 'a tensor of %s %s where the CPU has %s %s' % other))
            continue
        worst, nan = 0.0, 0
        for got, want in zip(result, reference):
            got = got.detach().float().cpu()
            nan += int(torch.isnan(got).sum()) + int(torch.isinf(got).sum())
            scale = float(want.abs().max()) or 1.0
            worst = max(worst, float((got - want).abs().max()) / scale)
        found.append((worst, nan, 'by %.2e of the largest value%s' % (worst, ', %d NaN or infinite' % nan if nan else '')))
    within = [worst is not None and worst <= tolerance and not nan for worst, nan, _ in found]
    if all(within):
        return True, 'from the CPU by %.2e of the largest value (allowed %.0e)' % (
            max(worst for worst, _, _ in found), tolerance)
    wrong = {(False, True): 'the first run differs', (True, False): 'the second run differs'}.get(
        tuple(within), 'both runs differ')
    return False, '%s from the CPU (the first run: %s; the second: %s; allowed %.0e)' % (
        wrong, found[0][2], found[1][2], tolerance)


def comparison_checks():
    """What verdict must refuse, and take, on tensors of the CPU: a check that was never seen to
    fail has not been shown to check. (Until 2026-11-23 the first run was not compared, and a
    run with fewer tensors than the CPU's was compared on those it had; and a tensor of another
    shape was broadcast to the CPU's: [1, 2] against [[1, 2], [1, 2]] was 0 off.)"""
    import torch
    right, off = torch.ones(4), torch.ones(4) + 1
    row, rows = torch.tensor([1.0, 2.0]), torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    wrong = []
    for runs, reference, ok, said in [
        ([row, row], rows, False, 'the first run: a tensor of (2,) torch.float32 where the CPU has (2, 2) torch.float32'),
        ([rows, rows[:1]], rows, False, 'the second: a tensor of (1, 2) torch.float32 where the CPU has (2, 2)'),
        ([right.double(), right], right, False, 'the first run: a tensor of (4,) torch.float64 where the CPU has (4,) torch.float32'),
        ([right, right], right, True, 'from the CPU by 0.00e+00'),
        ([(right, right), (right, right)], (right, right), True, 'from the CPU by 0.00e+00'),
        ([off, right], right, False, 'the first run differs'),
        ([right, off], right, False, 'the second run differs'),
        ([off, off], right, False, 'both runs differ'),
        ([right, right * float('nan')], right, False, 'the second: by 0.00e+00 of the largest value, 4 NaN or infinite'),
        ([(right,), (right,)], (right, right), False, 'the first run: 1 tensors where the CPU has 2'),
    ]:
        got, detail = verdict(runs, reference, 2e-3)
        if got != ok or said not in detail:
            wrong.append('%s (should %s, saying %r)' % (detail, 'pass' if ok else 'fail', said))
    return wrong


def stop(p):
    """Stop a case's process: SIGUSR1 first (its fault handler then writes where it was on its
    standard error), then SIGKILL. False if it has not ended even so."""
    for sig, wait in ((signal.SIGUSR1, 1), (signal.SIGKILL, 60)):
        p.send_signal(sig)
        try:
            p.wait(wait)
            return True
        except subprocess.TimeoutExpired:
            pass
    return False


def main():
    args = sys.argv[1:]
    if args[:1] == ['--worker']:
        return worker(args[1], args[2])
    if args[:1] == ['--list']:
        print('\n'.join(n for n, b, t in cases()))
        return 0
    if args[:1] == ['--comparison']:
        wrong = comparison_checks()
        for line in wrong:
            print('WRONG  the comparison: ' + line)
        print('FAIL  the comparison' if wrong else 'PASS  the comparison')
        return 1 if wrong else 0
    verbose = '-v' in args
    patterns = [a for a in args if a != '-v']
    wait = int(os.environ.get('VF_CASE_WAIT', '600'))
    names = [n for n, b, t in cases() if not patterns or any(p in n for p in patterns)]
    counts, bad = {}, False
    # (With the other logs when the script that runs this says where they are: not in /tmp,
    # which on the nodes is memory.)
    logs = os.path.join(os.environ['VF_OUT'], 'logs') if os.environ.get('VF_OUT') else None
    if logs:
        os.makedirs(logs, exist_ok=True)
    log = tempfile.NamedTemporaryFile(prefix='vf-models-', suffix='.log', delete=False, dir=logs).name
    answer = log[:-len('.log')] + '.verdict'
    env = dict(os.environ, VF_LOG=log, VF_LOG_LEVEL=os.environ.get('VF_LOG_LEVEL', '1'))
    for name in names:
        open(log, 'w').close()
        if os.path.exists(answer):
            os.unlink(answer)
        # (Standard output and error to files: the process is not held by a full pipe.)
        with tempfile.TemporaryFile(dir=logs) as out, tempfile.TemporaryFile(dir=logs) as err:
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), '--worker', name, answer],
                                 stdout=out, stderr=err, env=env)
            try:
                p.wait(wait)
                stopped = None
            except subprocess.TimeoutExpired:
                stopped = stop(p)
            err.seek(max(0, err.seek(0, 2) - 2000))
            tail = (err.read().decode(errors='replace').strip().splitlines() or [''])[-1][:160]
        verdict = open(answer, errors='replace').read() if os.path.exists(answer) else ''
        fields = verdict[:-1].split('\t', 1) if verdict.endswith('\n') and verdict.count('\n') == 1 else []
        if stopped is not None:
            detail = 'no end within %d s: stopped' % wait
            if verdict:
                detail += ' (it had written %r)' % verdict[:120]
            if not stopped:
                detail += '; it did not end even with SIGKILL: the run ends here'
            verdict = 'CRASH'
        elif p.returncode != 0 or not verdict:
            verdict, detail = 'CRASH', 'the process ended with status %d: %s' % (p.returncode, tail)
        elif len(fields) != 2 or fields[0] not in ('ok', 'DIFFERENT', 'refused', 'error'):
            verdict, detail = 'CRASH', 'the process ended with what is not a verdict: %r' % verdict[:160]
        else:
            verdict, detail = fields
        if verdict == 'refused':
            # What the proxy said of the kernels it did not give: their names, and why.
            said = []
            for line in open(log, errors='replace'):
                m = re.search(r'(\S+) is not given to the program: (.*)$', line)
                if m and m.group(0) not in said:
                    said.append(m.group(0))
            detail = '; '.join(re.sub(r'^_Z\d+', '', x)[:150] for x in said[:2]) or 'the proxy said nothing of it'
        counts[verdict] = counts.get(verdict, 0) + 1
        bad = bad or verdict in ('DIFFERENT', 'CRASH')
        if verbose or verdict != 'ok':
            print('%-9s %-28s %s' % (verdict, name, detail), flush=True)
        if stopped is False:
            break
    print('%d cases: %s%s' % (len(names), ', '.join('%d %s' % (counts[v], v) for v in ('ok', 'DIFFERENT', 'refused', 'error', 'CRASH') if v in counts),
                              ', %d not run' % (len(names) - sum(counts.values())) if sum(counts.values()) < len(names) else ''))
    os.unlink(log)
    if os.path.exists(answer):
        os.unlink(answer)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
