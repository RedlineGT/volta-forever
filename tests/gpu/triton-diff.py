#!/usr/bin/env python3
"""Triton kernels through the layer against a reference, and the two routes against each other.

The kernels and their cases are those of tests/gpu/triton_corpus.py (the generator; vendor
kernels are added by --vendor, see tests/gpu/triton_vendor.py). For each case:

  - the inputs: made once, from the case's seed, in one thread, by the command that makes the
    references, and saved beside them (DIR/inputs/CASE.pt, with their digest); the interpreter and
    both routes on the GPU take them from there. (Made again in each process, they could differ in
    the last bits: PyTorch's tanh on the CPU is MKL's, called by each thread of its pool on a block
    of 2,048 values, and in two processes of the Triton campaign one thread computed its block
    with MKL's AVX2 code of low accuracy instead of its AVX-512 code (ld-erfinv-f64: 890 to 902
    of the block's values one unit of the last place away, which erfinv near 1 magnifies); it
    also differs from one machine to another. Every record says the digest of the inputs it ran
    on, and the report says it with a difference.);
  - the reference: Triton's interpreter on the CPU (TRITON_INTERPRET=1, in a process without a
    GPU), or the case's own computation with PyTorch on the CPU where it names one;
  - the GPU, through the layer as the program is given it: Triton, told 7.5, makes sm_75 machine
    code, which the layer translates (route `sass`); each case is run --runs times on fresh copies
    of its inputs, and the runs must give the same bits;
  - the GPU through the layer's other route (route `ptx`): the same compiled kernel, but the module
    is loaded from the PTX that Triton made it from (the layer retargets it and the driver's
    compiler makes sm_70 code of it), with the same launcher and arguments;
  - the two routes must give the same bits (unless the case says why not), and each must agree
    with the reference: bit for bit, or within the tolerance the case states.

Commands:
  triton-diff.py interp --out DIR [--level L] [--filter RE] [--jobs N]
        the inputs and the reference of every case of the level, saved in DIR (one file per case)
  triton-diff.py inputs --out DIR [--level L] [--filter RE] [--jobs N]
        the inputs only; prints the digest of all of them (the same in every process)
  triton-diff.py gpu --route sass|ptx --ref DIR --out FILE [--runs N] [--level L] [--filter RE]
        [--shard I/N] [--case-wait S]
        run the cases on the GPU on the inputs of DIR, compare each with DIR, write one JSON line
        per case to FILE (the full outputs of a case that fails are saved beside FILE)
  triton-diff.py report FILE... [--json OUT]
        put the lines of both routes together: per case, the verdict
  triton-diff.py all --work DIR [--level quick] [--runs 5]
        the three above in order, as tests/gpu/triton.sh runs them; exit status 1 if a case fails

Verdicts: PASS; PASS-PTX (the same, but the layer refused the machine code and ran the PTX that the
cubin keeps: the translator was not tested); FAIL-REF (a route differs from the reference beyond the stated tolerance),
FAIL-DET (the runs of a route differ), FAIL-ROUTE (the routes give different bits where the case
says they must not), ERROR (a launch failed: the message says why; error 218 is a kernel the
layer refused), COMPILE (Triton's own compiler cannot make the kernel for sm_75: not the layer's),
CRASH (the process died or did not end in time), SKIP-REF (no reference for this case: the runs and
the routes were compared only).
"""
import argparse, hashlib, json, os, re, signal, subprocess, sys, time, traceback, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load_cases(level, filt, vendor):
    import triton_corpus
    cs = triton_corpus.cases(level)
    if vendor:
        import triton_vendor
        cs += triton_vendor.cases(level)
    if filt:
        r = re.compile(filt)
        cs = [c for c in cs if r.search(c.name)]
    return cs


def seed_of(name):
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


def make_inputs(case):
    """The case's inputs, from its seed, computed in one thread (in several, a thread of PyTorch's
    pool can take other code of MKL than the others: the head of this file)."""
    import torch
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        g = torch.Generator().manual_seed(seed_of(case.name))
        return case.make(g)
    finally:
        torch.set_num_threads(threads)


def raw(t):
    """The bytes of a CPU tensor, as a numpy array of uint8."""
    import torch
    t = t.contiguous().reshape(-1)
    if t.dtype == torch.bool:
        t = t.to(torch.uint8)
    return t.view(torch.uint8).numpy()


def input_digest(t):
    """The digest of a case's inputs: each entry in the order of the names, with its type, shape
    and bytes."""
    import torch
    h = hashlib.sha256()
    for k in sorted(t):
        v = t[k]
        if isinstance(v, torch.Tensor):
            h.update(f'{k} {v.dtype} {tuple(v.shape)}\n'.encode())
            if v.numel():
                h.update(raw(v))
        else:
            h.update(f'{k} {v!r}\n'.encode())
    return h.hexdigest()[:24]


# A tensor this large is saved compressed (one case of the full corpus has 2 GB, nearly all zeros).
PACKED = 64 << 20


def inputs_path(ref, case):
    return os.path.join(ref, 'inputs', case.name + '.pt')


def save_inputs(ref, case, t):
    """Save the inputs `t` of `case` in the directory `ref`; their digest."""
    import torch
    plain, packed = {}, {}
    for k, v in t.items():
        if isinstance(v, torch.Tensor) and v.numel() * v.element_size() >= PACKED:
            data = bytearray(zlib.compress(raw(v), 1))
            packed[k] = {'dtype': str(v.dtype).split('.')[-1], 'shape': list(v.shape),
                         'data': torch.frombuffer(data, dtype=torch.uint8)}
        else:
            plain[k] = v
    d = input_digest(t)
    path = inputs_path(ref, case)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'digest': d, 'plain': plain, 'packed': packed}, path + '.tmp')
    os.replace(path + '.tmp', path)
    return d


def load_inputs(ref, case):
    """(inputs, digest): the inputs of `case` saved in `ref`, checked against their digest."""
    import torch
    path = inputs_path(ref, case)
    if not os.path.exists(path):
        raise RuntimeError(f'no inputs in {path}: make them with `triton-diff.py interp` (or `inputs`)')
    d = torch.load(path, weights_only=True)
    t = dict(d['plain'])
    for k, p in d['packed'].items():
        data = bytearray(zlib.decompress(p['data'].numpy()))
        t[k] = torch.frombuffer(data, dtype=getattr(torch, p['dtype'])).reshape(p['shape'])
    got = input_digest(t)
    if got != d['digest']:
        raise RuntimeError(f'the inputs in {path} do not match their digest ({got}, saved {d["digest"]})')
    return t, got


def digest(t):
    import torch
    t = t.contiguous()
    if t.dtype == torch.bool:
        t = t.to(torch.uint8)
    b = t.view(torch.uint8) if t.numel() else torch.zeros(0, dtype=torch.uint8)
    return hashlib.sha256(b.numpy().tobytes()).hexdigest()[:24]


def apply_post(case, outs):
    return case.post(outs) if case.post else outs

# ------------------------------------------------------------------------------------------------
# The reference


def reference_of(case, inputs, interp_ok=True):
    """{name: CPU tensor} or None. Under the interpreter when the case's reference is it. `inputs`:
    the case's tensors (they are not changed)."""
    import torch
    t = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    if case.ref is None:
        return None
    if callable(case.ref):
        r = case.ref(t)
        return {k: r[k] for k in case.outs}
    if not interp_ok:
        raise RuntimeError('this process is not the interpreter')
    if case.ref == 'interp32':
        orig = {k: v.dtype for k, v in t.items() if isinstance(v, torch.Tensor)}
        t = {k: (v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16 else v) for k, v in t.items()}
        case.run(t, interpreted=True)
        return {k: (t[k].to(torch.bfloat16) if orig[k] == torch.bfloat16 else t[k]) for k in case.outs}
    case.run(t, interpreted=True)
    return {k: t[k] for k in case.outs}


def inputs_of(a, c):
    """(inputs, digest, made): the inputs of case `c` in the directory a.out, made and saved
    first when they are not there (or with --again)."""
    import torch
    if a.again or not os.path.exists(inputs_path(a.out, c)):
        t = make_inputs(c)
        return t, save_inputs(a.out, c, t), True
    d = torch.load(inputs_path(a.out, c), weights_only=True, mmap=True)
    return None, d['digest'], False


def cmd_inputs(a):
    """The inputs of every case, saved in a.out; prints the digest of all of them."""
    import torch
    torch.set_num_threads(1)
    cs = load_cases(a.level, a.filter, a.vendor)
    if a.shard:
        i, n = map(int, a.shard.split('/'))
        cs = [c for k, c in enumerate(cs) if k % n == i]
    whole = hashlib.sha256()
    made = 0
    for c in cs:
        _, d, new = inputs_of(a, c)
        whole.update(f'{c.name} {d}\n'.encode())
        made += new
    print(f'inputs: {made} made, {len(cs)} cases, digest of all {whole.hexdigest()[:24]}')
    return 0


def cmd_interp(a):
    """In a process of its own, with TRITON_INTERPRET=1 and no GPU."""
    assert os.environ.get('TRITON_INTERPRET') == '1', 'run with TRITON_INTERPRET=1'
    import torch
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    cs = load_cases(a.level, a.filter, a.vendor)
    if a.shard:
        i, n = map(int, a.shard.split('/'))
        cs = [c for k, c in enumerate(cs) if k % n == i]
    done = 0
    for c in cs:
        path = os.path.join(a.out, c.name + '.pt')
        t, d, made = inputs_of(a, c)
        if c.ref is None or callable(c.ref):
            continue
        if os.path.exists(path) and not made:
            continue
        if t is None:
            t, d = load_inputs(a.out, c)
        start = time.time()
        try:
            r = reference_of(c, t)
            torch.save({'outs': r, 'time': time.time() - start, 'inputs': d}, path + '.tmp')
        except Exception as e:
            torch.save({'error': f'{type(e).__name__}: {e}'[:2000], 'time': time.time() - start, 'inputs': d},
                       path + '.tmp')
        os.replace(path + '.tmp', path)
        done += 1
    print(f'interp: {done} references made, {len(cs)} cases')
    return 0


def cmd_interp_parallel(a):
    """Run `interp` (or `inputs`) in --jobs processes (each with a shard of the cases)."""
    env = dict(os.environ, TRITON_INTERPRET='1', CUDA_VISIBLE_DEVICES='')
    for name in ('LD_AUDIT', 'LD_PRELOAD'):
        env.pop(name, None)
    base = [sys.executable, '-s', os.path.abspath(__file__), a.cmd, '--out', a.out, '--level', a.level]
    if a.filter:
        base += ['--filter', a.filter]
    if a.vendor:
        base += ['--vendor']
    if a.again:
        base += ['--again']
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    procs = [subprocess.Popen(base + ['--shard', f'{i}/{a.jobs}'], env=env) for i in range(a.jobs)]
    status = max(p.wait() for p in procs)
    if a.cmd == 'inputs' and status == 0:
        # the digest of all of them, from the files the shards made
        return cmd_inputs(argparse.Namespace(**dict(vars(a), shard='', again=False)))
    return status

# ------------------------------------------------------------------------------------------------
# The GPU


def use_ptx_route():
    """Load every module Triton makes from its PTX instead of its cubin: the layer's other route."""
    from triton.compiler import compiler as C
    orig = C.CompiledKernel._init_handles

    def init(self):
        if self.module is None and 'ptx' in self.asm:
            self.kernel = self.asm['ptx'].encode() + b'\0'
        return orig(self)
    C.CompiledKernel._init_handles = init


def compare(case, got, ref):
    """(ok, detail): got against ref, by the case's rule."""
    import torch
    worst = []
    ok = True
    for k in case.outs:
        g, r = got[k], ref[k]
        if g.shape != r.shape or g.dtype != r.dtype:
            return False, f'{k}: shape/dtype {tuple(g.shape)}/{g.dtype} against {tuple(r.shape)}/{r.dtype}'
        if g.dtype.is_floating_point:
            gn, rn = g.isnan(), r.isnan()
            gb = g.view(torch.int16 if g.element_size() == 2 else torch.int32 if g.element_size() == 4 else torch.int64)
            rb = r.view(gb.dtype)
            same = (gb == rb) | (gn & rn)
            if case.cmp == 'exact0':
                # bit for bit, but the two zeros are one
                same = same | ((g == 0) & (r == 0))
            if case.cmp in ('exact', 'exact0'):
                bad = ~same
            else:
                _, rtol, atol = case.cmp
                gd, rd = g.double(), r.double()
                close = (gd - rd).abs() <= atol + rtol * rd.abs()
                # infinities: the same infinity; NaN: NaN on both sides
                close = close | same
                close = close & ~(gn ^ rn) & ~(g.isinf() & (gd != rd))
                bad = ~close
        else:
            bad = g != r
        nbad = int(bad.sum())
        if nbad:
            ok = False
            idx = bad.reshape(-1).nonzero()[:3].reshape(-1).tolist()
            gf, rf = g.reshape(-1), r.reshape(-1)
            ex = ', '.join(f'[{i}] {gf[i].item()!r} vs {rf[i].item()!r}' for i in idx)
            if g.dtype.is_floating_point:
                d = (g.double() - r.double()).abs()
                fin = ~(g.isnan() | r.isnan() | g.isinf() | r.isinf())
                md = float(d[fin].max()) if bool(fin.any()) else float('nan')
                worst.append(f'{k}: {nbad} of {g.numel()} differ (max |diff| {md:.3g}); {ex}')
            else:
                worst.append(f'{k}: {nbad} of {g.numel()} differ; {ex}')
    return ok, '; '.join(worst)


class VfLog:
    """The lines the layer wrote to its log (VF_LOG) since the last call."""

    def __init__(self):
        self.path = os.environ.get('VF_LOG')
        self.pos = os.path.getsize(self.path) if self.path and os.path.exists(self.path) else 0

    def new(self):
        if not self.path or not os.path.exists(self.path):
            return []
        with open(self.path, 'rb') as f:
            f.seek(self.pos)
            data = f.read()
        self.pos += len(data)
        return [l for l in data.decode(errors='replace').splitlines() if l.strip()]


def route_of(lines):
    """What the layer said of the modules loaded (Triton loads its modules with cuModuleLoadData;
    PyTorch's libraries, loaded meanwhile with cuLibraryLoadData, are not counted)."""
    kinds = set()
    for l in lines:
        module = 'cuModuleLoadData' in l
        if module and ('machine code for sm_75 translated' in l or 'machine code for sm_80 translated' in l):
            kinds.add('translated')
        if module and 'retargeted' in l:
            kinds.add('ptx')
        if 'cannot run on this GPU' in l:
            kinds.add('refused')
        if 'emulated in' in l:
            kinds.add('emulated')
    return sorted(kinds)


def cmd_worker(a):
    """Run the cases from --start on, appending one JSON line per case to --out (a line 'start'
    before each, so that the supervisor knows which one a crash was in)."""
    import torch
    if a.route == 'ptx':
        use_ptx_route()
    cs = load_cases(a.level, a.filter, a.vendor)
    if a.shard:
        i, n = map(int, a.shard.split('/'))
        cs = [c for k, c in enumerate(cs) if k % n == i]
    log = VfLog()
    faildir = a.out + '.fail'
    out = open(a.out, 'a')

    def emit(rec):
        out.write(json.dumps(rec) + '\n')
        out.flush()
    for idx in range(a.start, len(cs)):
        c = cs[idx]
        emit({'start': c.name, 'idx': idx})
        rec = {'case': c.name, 'route': a.route, 'idx': idx}
        t0 = time.time()
        try:
            # The inputs that the reference was made from, the same for both routes (made again here,
            # they could differ in the last bits: the head of this file says why).
            inputs, rec['inputs'] = load_inputs(a.ref, c)
            ref = None
            if c.ref is not None:
                if callable(c.ref):
                    ref = reference_of(c, inputs, interp_ok=False)
                else:
                    p = os.path.join(a.ref, c.name + '.pt')
                    if os.path.exists(p):
                        d = torch.load(p)
                        if d.get('inputs') != rec['inputs']:
                            raise RuntimeError(f'the reference {p} was made from other inputs ({d.get("inputs")})')
                        ref = d.get('outs')
                        if ref is None:
                            rec['ref_error'] = d.get('error')
                    else:
                        rec['ref_error'] = 'no reference file'
            hashes, first, det_detail = [], None, ''
            for run in range(a.runs):
                # (A fresh copy of the same inputs for every run.)
                td = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
                c.run(td)
                torch.cuda.synchronize()
                outs = apply_post(c, {k: td[k].cpu() for k in c.outs})
                h = {k: digest(v) for k, v in outs.items()}
                hashes.append(h)
                if first is None:
                    first = outs
                elif h != hashes[0] and not det_detail:
                    _, det_detail = compare(type('E', (), {'outs': c.outs, 'cmp': 'exact'})(), outs, first)
                    det_detail = f'run {run}: ' + det_detail
                del td
            rec['hash'] = hashes[0]
            rec['det'] = all(h == hashes[0] for h in hashes) or c.nondet
            rec['ndistinct'] = len({json.dumps(h, sort_keys=True) for h in hashes})
            if det_detail:
                rec['det_detail'] = det_detail
            if ref is not None:
                ok, detail = compare(c, first, apply_post(c, ref))
                rec['ref_ok'] = ok
                if detail:
                    rec['ref_detail'] = detail
            else:
                rec['ref_ok'] = None
            if rec.get('ref_ok') is False or not rec['det']:
                os.makedirs(faildir, exist_ok=True)
                torch.save({'got': first, 'ref': ref}, os.path.join(faildir, f'{c.name}.{a.route}.pt'))
            rec['status'] = 'ok'
        except Exception as e:
            msg = f'{type(e).__name__}: {e}'
            rec['status'] = 'error'
            rec['error'] = msg[:3000]
            rec['tb'] = traceback.format_exc()[-3000:]
        rec['time'] = round(time.time() - t0, 3)
        lines = log.new()
        rec['vf'] = route_of(lines)
        rec['vflog'] = [l[:400] for l in lines[:12]]
        emit(rec)
        if rec['status'] == 'error' and sticky(rec['error']):
            # the context is lost: the supervisor starts another process from the next case
            out.close()
            os._exit(3)
    out.close()
    return 0


def sticky(msg):
    return any(s in msg for s in ('illegal address', 'misaligned address', 'unspecified launch failure',
                                  'illegal instruction', 'an illegal memory access', 'launch timed out',
                                  'CUDA_ERROR_ILLEGAL', 'CUDA_ERROR_LAUNCH_FAILED', 'CUDA_ERROR_MISALIGNED',
                                  'device-side assert', 'hardware stack error', 'CUDA error: an illegal'))


def cmd_gpu(a):
    """Supervise workers: restart after a crash or a stall, from the case after it."""
    cs = load_cases(a.level, a.filter, a.vendor)
    if a.shard:
        i, n = map(int, a.shard.split('/'))
        cs = [c for k, c in enumerate(cs) if k % n == i]
    total = len(cs)
    done_names = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if 'case' in r:
                done_names.add(r['case'])
    start = 0
    while start < total and cs[start].name in done_names:
        start += 1
    base = [sys.executable, '-s', os.path.abspath(__file__), 'worker', '--route', a.route, '--ref', a.ref,
            '--out', a.out, '--runs', str(a.runs), '--level', a.level]
    if a.filter:
        base += ['--filter', a.filter]
    if a.shard:
        base += ['--shard', a.shard]
    if a.vendor:
        base += ['--vendor']
    while start < total:
        p = subprocess.Popen(base + ['--start', str(start)])
        last_size, last_change = -1, time.time()
        while p.poll() is None:
            time.sleep(1)
            size = os.path.getsize(a.out) if os.path.exists(a.out) else 0
            if size != last_size:
                last_size, last_change = size, time.time()
            elif time.time() - last_change > a.case_wait:
                p.send_signal(signal.SIGKILL)
                p.wait()
                break
        # which case was the process in
        started, finished = None, set()
        for l in open(a.out):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if 'start' in r:
                started = r
            elif 'case' in r:
                finished.add(r['case'])
        if started is None or started['start'] in finished:
            nxt = max([k for k, c in enumerate(cs) if c.name in finished] + [start - 1]) + 1
            if p.returncode not in (0, None) and nxt == start:
                # died before it started anything
                with open(a.out, 'a') as f:
                    f.write(json.dumps({'case': cs[start].name, 'route': a.route, 'status': 'crash',
                                        'error': f'worker exit {p.returncode} before the case'}) + '\n')
                nxt = start + 1
            start = nxt
            if p.returncode == 0:
                break
            continue
        with open(a.out, 'a') as f:
            f.write(json.dumps({'case': started['start'], 'route': a.route, 'status': 'crash', 'idx': started['idx'],
                                'error': f'worker exit {p.returncode} (killed after {a.case_wait} s without progress)'
                                if p.returncode == -9 else f'worker exit {p.returncode}'}) + '\n')
        start = started['idx'] + 1
    return 0

# ------------------------------------------------------------------------------------------------
# Verdicts


def verdicts(files, cases_by_name=None):
    recs = {}
    for fn in files:
        for l in open(fn):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if 'case' not in r:
                continue
            recs.setdefault(r['case'], {})[r['route']] = r
    out = {}
    for name, by in sorted(recs.items()):
        c = cases_by_name.get(name) if cases_by_name else None
        v, why = 'PASS', []
        for route in ('sass', 'ptx'):
            r = by.get(route)
            if r is None:
                continue
            if r['status'] == 'crash':
                v = 'CRASH'
                why.append(f'{route}: {r.get("error")}')
                continue
            if r['status'] == 'error':
                compile_error = any(w in r.get('tb', '') + r.get('error', '') for w in (
                    'triton/compiler', 'CompilationError', 'PTXASError', 'PassManager::run failed'))
                # (Not the limit of threads that Triton reads from the loaded kernel: the register
                # count of the translation sets it, the layer's doing.)
                compile_error = compile_error and 'out of resource: threads' not in r.get('error', '')
                if v in ('PASS', 'COMPILE'):
                    # Triton's own compiler refused the kernel for this target: not the layer's
                    v = 'COMPILE' if compile_error else 'ERROR'
                why.append(f'{route}: {r.get("error", "")[:300]}')
                continue
            if not r.get('det', True):
                v = 'FAIL-DET'
                why.append(f'{route}: {r.get("ndistinct")} different results in the runs on inputs {r.get("inputs")}; '
                           f'{r.get("det_detail", "")[:300]}')
            if r.get('ref_ok') is False:
                if v in ('PASS', 'SKIP-REF', 'ERROR'):
                    v = 'FAIL-REF'
                why.append(f'{route}: on inputs {r.get("inputs")}: {r.get("ref_detail", "")[:400]}')
            if r.get('ref_ok') is None and v == 'PASS':
                v = 'SKIP-REF'
                if r.get('ref_error'):
                    why.append(f'{route}: no reference: {r["ref_error"][:200]}')
        s, p = by.get('sass'), by.get('ptx')
        if s and p and s.get('status') == p.get('status') == 'ok' and s.get('inputs') != p.get('inputs'):
            # (Not the layer's: the inputs were made again between the two routes.)
            if v in ('PASS', 'SKIP-REF'):
                v = 'FAIL-ROUTE'
            why.append(f'the routes ran on different inputs (sass {s.get("inputs")}, ptx {p.get("inputs")})')
        elif s and p and s.get('status') == p.get('status') == 'ok' and s.get('hash') != p.get('hash'):
            exact = c.route_exact if c is not None else True
            if exact and v in ('PASS', 'SKIP-REF'):
                v = 'FAIL-ROUTE'
            why.append(f'the routes give different bits on the same inputs ({s.get("inputs")})'
                       + ('' if exact else ' (allowed by the case)'))
        # The route each module took, by the layer's lines (when the case loaded a module: one of an
        # earlier case's is not loaded again)
        if s and s.get('status') == 'ok' and 'ptx' in s.get('vf', []):
            # every function of the cubin refused by the translator: the layer ran the PTX that
            # Triton keeps in it (right, but the machine code was not tested)
            if v == 'PASS':
                v = 'PASS-PTX'
            why.append('sass route: the machine code was refused and the layer ran the PTX kept in the cubin')
        if p and p.get('status') == 'ok' and 'translated' in p.get('vf', []):
            if v in ('PASS', 'SKIP-REF'):
                v = 'FAIL-ROUTE'
            why.append(f'ptx route: the layer said {p.get("vf")}: a module was not retargeted PTX')
        out[name] = {'verdict': v, 'why': why, 'vf': {k: by[k].get('vf') for k in by},
                     'time': {k: by[k].get('time') for k in by}, 'inputs': {k: by[k].get('inputs') for k in by}}
    return out


def cmd_report(a):
    cases_by_name = None
    try:
        cases_by_name = {c.name: c for c in load_cases(a.level, a.filter, a.vendor)}
    except Exception:
        pass
    v = verdicts(a.files, cases_by_name)
    counts = {}
    for name, r in v.items():
        counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
        if r['verdict'] not in ('PASS',) or a.all:
            print(f'{r["verdict"]:10s} {name}: ' + ' | '.join(r['why']).replace('\n', ' '))
    print('summary: ' + ', '.join(f'{k} {n}' for k, n in sorted(counts.items())) + f' (of {len(v)} cases)')
    if a.json:
        json.dump(v, open(a.json, 'w'), indent=1)
    bad = sum(n for k, n in counts.items() if k not in ('PASS', 'PASS-PTX', 'SKIP-REF', 'COMPILE'))
    return 1 if bad else 0


def cmd_all(a):
    """What tests/gpu/triton.sh runs: reference, both routes, report."""
    os.makedirs(a.work, exist_ok=True)
    ref = os.path.join(a.work, 'ref')
    env = dict(os.environ, TRITON_INTERPRET='1', CUDA_VISIBLE_DEVICES='')
    for name in ('LD_AUDIT', 'LD_PRELOAD'):
        env.pop(name, None)
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    common = ['--level', a.level] + (['--filter', a.filter] if a.filter else []) + (['--vendor'] if a.vendor else [])
    t0 = time.time()
    procs = [subprocess.Popen([sys.executable, '-s', os.path.abspath(__file__), 'interp', '--out', ref,
                               '--shard', f'{i}/{a.jobs}'] + common, env=env) for i in range(a.jobs)]
    if any(p.wait() for p in procs):
        print('FAIL: the interpreter could not run')
        return 1
    print(f'references: {time.time() - t0:.0f} s')
    files = []
    for route in ('sass', 'ptx'):
        t0 = time.time()
        f = os.path.join(a.work, f'{route}.jsonl')
        if os.path.exists(f):
            os.remove(f)
        r = subprocess.run([sys.executable, '-s', os.path.abspath(__file__), 'gpu', '--route', route, '--ref', ref,
                            '--out', f, '--runs', str(a.runs if route == 'sass' else max(2, a.runs // 2)),
                            '--case-wait', str(a.case_wait)] + common)
        files.append(f)
        print(f'route {route}: {time.time() - t0:.0f} s')
    ns = argparse.Namespace(files=files, json=os.path.join(a.work, 'verdicts.json'), all=False, level=a.level,
                            filter=a.filter, vendor=a.vendor)
    return cmd_report(ns)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('--level', default='full', choices=['quick', 'full'])
        p.add_argument('--filter', default='')
        p.add_argument('--vendor', action='store_true', help='also the kernels of vLLM and SGLang (triton_vendor.py)')
    for name in ('interp', 'inputs'):
        p = sub.add_parser(name)
        common(p)
        p.add_argument('--out', required=True)
        p.add_argument('--shard', default='')
        p.add_argument('--jobs', type=int, default=0, help='run in N processes')
        p.add_argument('--again', action='store_true')
    p = sub.add_parser('worker')
    common(p)
    p.add_argument('--route', required=True, choices=['sass', 'ptx'])
    p.add_argument('--ref', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--shard', default='')
    p = sub.add_parser('gpu')
    common(p)
    p.add_argument('--route', required=True, choices=['sass', 'ptx'])
    p.add_argument('--ref', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--shard', default='')
    p.add_argument('--case-wait', type=int, default=600)
    p = sub.add_parser('report')
    common(p)
    p.add_argument('files', nargs='+')
    p.add_argument('--json', default='')
    p.add_argument('--all', action='store_true')
    p = sub.add_parser('all')
    common(p)
    p.add_argument('--work', required=True)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--case-wait', type=int, default=600)
    a = ap.parse_args()
    if a.cmd in ('interp', 'inputs'):
        return cmd_interp_parallel(a) if a.jobs else {'interp': cmd_interp, 'inputs': cmd_inputs}[a.cmd](a)
    return {'worker': cmd_worker, 'gpu': cmd_gpu, 'report': cmd_report, 'all': cmd_all}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
