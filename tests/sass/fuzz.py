#!/usr/bin/env python3
"""The Rust translator (crates/vf-sass) against the Python reference (reference/translate.py) on
damaged cubins: what compare.sh cannot show, because real code never takes those paths.

    tests/sass/fuzz.py [-n COUNT] [-s SEED] [-j JOBS] [--pool FILE] [--keep DIR] CUBIN...
    tests/sass/fuzz.py --make-pool FILE CUBIN...

Each round takes one of the cubins (small ones: the prototype translates every mutant), damages it,
and gives it to both translators. They must agree on everything: the bytes of the result, the map of
its instructions, which functions are refused and why. Where the prototype stops on an exception
(it was written for well-formed files), the translator must still end properly, and the place is
counted: each such place is to be looked at once.

The damage, one or two of these per round:
  splice    instructions replaced by others taken from real code (the pool: `--make-pool` gathers,
            from any cubins, a few instructions of every class they hold)
  field     one operand, modifier or constant field of an instruction set to another value
  bits      one or two bits of an instruction flipped
  schedule  the stall count, the scoreboards or the wait mask of an instruction changed
  stalls    the longest stall on a run of instructions, with no scoreboard: a wait that the
            translation needs there no longer fits in a stall count
  swap, repeat, delete
            instructions exchanged, doubled or removed (the last two move what follows)
  info      a byte of the attributes of a function, or of the file, changed
  registers the register count of a function changed
  relocation
            a relocation changed: where it applies, its kind, the symbol it names (which may then
            be a function), or one of its bytes
  container a byte of the headers changed, or the file cut short

A mutant on which they differ is kept in the directory `--keep` names, with a line saying how it
was made. The exit status is 0 when none differs and the translator never ended badly.
"""
import argparse, collections, hashlib, multiprocessing, os, random, re, shutil, struct, subprocess, sys, tempfile, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '../../reference'))

KINDS = [('splice', 30), ('field', 30), ('bits', 8), ('schedule', 6), ('stalls', 6), ('swap', 5), ('repeat', 4),
         ('delete', 4), ('info', 6), ('registers', 2), ('container', 5), ('relocation', 6)]


def words_of(data):
    return [int.from_bytes(data[i:i + 16], 'little') for i in range(0, len(data) - 15, 16)]


def code_of(words):
    return b''.join(w.to_bytes(16, 'little') for w in words)


def damage(rng, mc, elf, data, pool):
    """`data` damaged in one way: (bytes, what was done)."""
    kind = rng.choices([k for k, _ in KINDS], [w for _, w in KINDS])[0]
    if kind == 'container':
        out = bytearray(data)
        if rng.random() < 0.3:
            at = rng.randrange(1, len(out))
            return bytes(out[:at]), 'container: cut at %d of %d' % (at, len(data))
        shoff, = struct.unpack_from('<Q', data, 0x28)
        shnum, = struct.unpack_from('<H', data, 0x3c)
        where = rng.choice(['header', 'sections', 'sections', 'sections'])
        at = rng.randrange(0, 64) if where == 'header' else shoff + rng.randrange(0, 64 * max(1, shnum))
        if at < len(out):
            out[at] ^= 1 << rng.randrange(8)
        return bytes(out), 'container: a bit of byte 0x%x (%s)' % (at, where)
    e = elf.Elf(data)
    texts = [s for s in e.sections if s.name.startswith('.text.') and s.type == 1 and len(s.data) >= 32]
    if kind == 'info':
        infos = [s for s in e.sections if s.name.startswith('.nv.info') and len(s.data) > 4]
        if infos:
            s = rng.choice(infos)
            d = bytearray(s.data)
            at = rng.randrange(len(d))
            d[at] = rng.choice([d[at] ^ (1 << rng.randrange(8)), rng.randrange(256), 0, 0xff])
            s.data = bytes(d)
            return e.write(), 'info: byte %d of %s' % (at, s.name[:40])
        kind = 'bits'
    if kind == 'relocation':
        relocations = [s for s in e.sections if s.type in (4, 9) and len(s.data) >= 16]
        if relocations:
            s = rng.choice(relocations)
            step = 24 if s.type == 4 else 16
            d = bytearray(s.data)
            at = step * rng.randrange(max(1, len(d) // step))
            offset, info = struct.unpack_from('<QQ', d, at)
            what = rng.choice(['offset', 'offset', 'kind', 'symbol', 'byte'])
            if what == 'offset':
                offset = rng.choice([offset + 16, max(0, offset - 16), offset + 8, offset ^ (1 << rng.randrange(20)), 1 << 40])
            elif what == 'kind':
                info = (info & ~0xffffffff) | rng.choice([0, 1, 2, 55, 56, 57, 58, 64, rng.randrange(256)])
            elif what == 'symbol':
                info = (rng.randrange(200) << 32) | (info & 0xffffffff)
            struct.pack_into('<QQ', d, at, offset, info)
            if what == 'byte':
                d[at + rng.randrange(step)] ^= 1 << rng.randrange(8)
            s.data = bytes(d)
            return e.write(), 'relocation: the %s of entry %d of %s' % (what, at // step, s.name[:40])
        kind = 'bits'
    if not texts:
        return data, 'nothing'
    s = rng.choice(texts)
    words = words_of(s.data)
    n = len(words)
    i = rng.randrange(n)
    name = s.name[len('.text.'):][:40]
    if kind == 'registers':
        count = rng.choice([0, 1, (s.info >> 24) - 1, (s.info >> 24) + 1, 250, 255, rng.randrange(256)]) & 0xff
        s.info = (count << 24) | (s.info & 0xffffff)
        return e.write(), 'registers: %s declares %d' % (name, count)
    if kind == 'splice':
        places = sorted({rng.randrange(n) for _ in range(rng.choice([1, 1, 1, 2, 3]))})
        for at in places:
            words[at] = rng.choice(pool)
        what = 'splice: %s at %s' % (name, ','.join('0x%x' % (16 * at) for at in places))
    elif kind == 'field':
        d = mc.decode(75, words[i])
        steps = [(k, f) for k, fields, _ in mc.d75.plan(d.c) for f in fields if k != 'unused'] if d is not None else []
        if not steps:
            words[i] ^= 1 << rng.randrange(128)
            what = 'bits: %s at 0x%x (no class)' % (name, 16 * i)
        else:
            k, f = rng.choice(steps)
            top = (1 << len(f)) - 1
            other = f.get(rng.choice(words))
            value = rng.choice([0, 1, top, top - 1, rng.randrange(top + 1), rng.randrange(top + 1), other, other])
            words[i] = f.put(words[i], value)
            what = 'field: %s at 0x%x, %s (%s) = %d' % (name, 16 * i, f.name, k, value)
    elif kind == 'bits':
        for _ in range(rng.choice([1, 1, 2])):
            words[i] ^= 1 << rng.randrange(128)
        what = 'bits: %s at 0x%x' % (name, 16 * i)
    elif kind == 'schedule':
        words[i] = (words[i] & ~(0xfffff << 105)) | (rng.choice([rng.randrange(1 << 20), (words[i] >> 105 ^ 1 << rng.randrange(20)) & 0xfffff]) << 105)
        what = 'schedule: %s at 0x%x' % (name, 16 * i)
    elif kind == 'stalls':
        last = min(n, i + rng.choice([2, 3, 4, 8, 16, 40]))
        for at in range(i, last):
            # A stall of 15, nothing announced on a scoreboard (7), the wait mask as it was.
            words[at] = (words[at] & ~(0x7ff << 105)) | ((15 | (7 << 5) | (rng.choice([7, 7, (words[at] >> 113) & 7]) << 8)) << 105)
        what = 'stalls: %s from 0x%x to 0x%x' % (name, 16 * i, 16 * last)
    elif kind == 'swap':
        j = rng.randrange(n)
        words[i], words[j] = words[j], words[i]
        what = 'swap: %s 0x%x and 0x%x' % (name, 16 * i, 16 * j)
    elif kind == 'repeat':
        words.insert(i, words[i])
        what = 'repeat: %s at 0x%x' % (name, 16 * i)
    else:
        del words[i]
        what = 'delete: %s at 0x%x' % (name, 16 * i)
    s.data = code_of(words)
    return e.write(), what


def mutant(rng, mc, elf, data, pool):
    out, what = damage(rng, mc, elf, data, pool)
    if rng.random() < 0.2:
        try:
            out, more = damage(rng, mc, elf, out, pool)
            what += ' + ' + more
        except Exception:
            pass                        # the first damage made it unreadable to the prototype's container
    return out, what


def prototype(mc, translate, data):
    """What the prototype makes of `data`: (bytes, map text, report text), or the place it stopped."""
    try:
        listing = []
        out, report = translate.translate(data, mc, listing)
    except Exception as e:
        frame = traceback.extract_tb(e.__traceback__)[-1]
        return None, '%s at %s:%d (%s)' % (type(e).__name__, os.path.basename(frame.filename), frame.lineno, (frame.line or '')[:60])
    listing = ''.join('%s\t%x\t%x\t%s\n' % row for row in listing)
    report = ''.join('%s\t%s\n' % (why or 'ok', function) for function, why in report.items())
    return (out, listing, report), None


def rounds(job):
    """One worker's share: `count` mutants from `seed`."""
    seed, count, paths, pool_path, tool, keep, tmp = job
    import elf, machine, translate
    mc = machine.Machine()
    rng = random.Random(seed)
    pool = [int(line, 16) for line in open(pool_path)] if pool_path else []
    seeds, kept_paths = [], []
    for path in paths:
        data = open(path, 'rb').read()
        own = [w for s in elf.Elf(data).sections if s.name.startswith('.text.') and s.type == 1 for w in words_of(s.data)]
        if len(own) >= 2:               # a cubin without code has nothing to damage
            seeds.append(data)
            kept_paths.append(path)
            pool += own
    paths = kept_paths
    tally, reasons, stops, kept = collections.Counter(), collections.Counter(), collections.Counter(), []
    work = tempfile.mkdtemp(dir=tmp)
    paths_ = [os.path.join(work, n) for n in ('in', 'out', 'map', 'report')]
    for k in range(count):
        which = rng.randrange(len(seeds))
        try:
            data, what = mutant(rng, mc, elf, seeds[which], pool)
        except Exception:
            tally[('(none)', 'the damage itself failed')] += 1
            continue
        kind = what.split(':')[0]
        ours, stop = prototype(mc, translate, data)
        for p in paths_[1:]:
            if os.path.exists(p):
                os.remove(p)
        open(paths_[0], 'wb').write(data)
        try:
            r = subprocess.run([tool, 'translate'] + paths_, capture_output=True, timeout=300)
            code, said = r.returncode, r.stderr.decode(errors='replace').strip()[:300]
        except subprocess.TimeoutExpired:
            code, said = 'timeout', ''
        read = lambda p, mode: open(p, mode).read() if os.path.exists(p) else None
        theirs = (read(paths_[1], 'rb'), read(paths_[2], 'r'), read(paths_[3], 'r'))
        if code not in (0, 1):
            verdict = 'ENDED BADLY (%s)' % code
        elif ours is None:
            stops[stop] += 1
            verdict = 'prototype stopped, translator %s' % ('refused the file' if code == 1 else 'translated')
        elif code == 1:
            verdict = 'DIFFERENT: the translator refused the file'
        else:
            parts = [n for n, a, b in zip(('bytes', 'map', 'refusals'), ours, theirs) if a != b]
            verdict = 'DIFFERENT: ' + ', '.join(parts) if parts else 'same'
            for line in ours[2].splitlines():
                reasons[re.sub(r'0x-?[0-9a-f]+|\b[0-9a-f]{32}\b|\bU?[RP]\d+\b|\d+', 'N', line.split('\t')[0])[:90]] += 1
        tally[(kind, verdict)] += 1
        if verdict.startswith(('DIFFERENT', 'ENDED')) or (ours is None and code == 0):
            note = '%s | seed %d round %d | %s | %s | %s' % (verdict, seed, k, os.path.basename(paths[which]), what, stop or said)
            if keep and len(kept) < 40:
                name = 'mutant-%d-%d.cubin' % (seed, k)
                open(os.path.join(keep, name), 'wb').write(data)
                note = name + ' | ' + note
            kept.append(note)
    shutil.rmtree(work, ignore_errors=True)
    return tally, reasons, stops, kept


def make_pool(out, paths, per_class=24):
    """A few instructions of every sm_75 class that the cubins hold, one hexadecimal word a line."""
    import elf, machine
    mc = machine.Machine()
    seen, by_class = set(), collections.defaultdict(list)
    rng = random.Random(1)
    for path in paths:
        e = elf.Elf(open(path, 'rb').read())
        for s in e.sections:
            if s.name.startswith('.text.') and s.type == 1:
                for w in words_of(s.data):
                    key = w & ~(0xfffff << 105)          # the same instruction scheduled otherwise is the same
                    if key in seen:
                        continue
                    seen.add(key)
                    d = mc.decode(75, w)
                    name = d.c.name if d is not None else '(none)'
                    if len(by_class[name]) < per_class:
                        by_class[name].append(w)
                    elif rng.random() < 0.02:
                        by_class[name][rng.randrange(per_class)] = w
    with open(out, 'w') as f:
        for name in sorted(by_class):
            for w in by_class[name]:
                f.write('%032x\n' % w)
    print('%d instructions of %d classes from %d cubins' % (sum(map(len, by_class.values())), len(by_class), len(paths)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('-n', type=int, default=2000, help='mutants (default 2000)')
    ap.add_argument('-s', type=int, default=1, help='seed (default 1)')
    ap.add_argument('-j', type=int, default=min(48, os.cpu_count() or 1), help='processes')
    ap.add_argument('--pool', help='instructions to splice in, besides those of the cubins')
    ap.add_argument('--keep', help='directory for the mutants on which the two differ')
    ap.add_argument('--tmp', help='directory for temporary files')
    # (The tool: as tests/sass/compare.sh finds it.)
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../target')
    # (Where scripts/build.sh builds this checkout: scripts/target-dir.sh.)
    checkout = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
    own = 'volta-forever/target-' + hashlib.sha256(checkout.encode()).hexdigest()[:12]
    built = os.environ.get('CARGO_TARGET_DIR') or (here if os.access(os.path.join(here, 'release/vf-translate'), os.X_OK) else
                                                   os.path.join(os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache'), own))
    ap.add_argument('--tool', default=os.environ.get('VF_TOOL') or os.path.normpath(os.path.join(built, 'release/vf-translate')))
    ap.add_argument('--make-pool', metavar='FILE')
    ap.add_argument('cubins', nargs='+')
    a = ap.parse_args()
    if a.make_pool:
        return make_pool(a.make_pool, a.cubins)
    if a.keep:
        os.makedirs(a.keep, exist_ok=True)
    share = max(1, a.n // a.j)
    jobs = [(a.s * 100003 + w, share, a.cubins, a.pool, a.tool, a.keep, a.tmp) for w in range(max(1, a.n // share))]
    tally, reasons, stops, kept = collections.Counter(), collections.Counter(), collections.Counter(), []
    with multiprocessing.Pool(a.j) as pool:
        for t, r, s, k in pool.imap_unordered(rounds, jobs):
            tally += t
            reasons += r
            stops += s
            kept += k
    total = sum(tally.values())
    verdicts = collections.Counter()
    for (kind, verdict), n in tally.items():
        verdicts[verdict] += n
    print('%d mutants of %d cubins' % (total, len(a.cubins)))
    for verdict, n in verdicts.most_common():
        print('  %7d  %s' % (n, verdict))
    kinds = sorted({kind for kind, _ in tally})
    print('by damage (same / prototype stopped / different or ended badly):')
    for kind in kinds:
        same = tally[(kind, 'same')]
        stopped = sum(n for (k, v), n in tally.items() if k == kind and v.startswith('prototype stopped'))
        bad = sum(n for (k, v), n in tally.items() if k == kind and v.startswith(('DIFFERENT', 'ENDED')))
        print('  %-10s %7d %7d %7d' % (kind, same, stopped, bad))
    print('what the functions of the mutants came to (the prototype), most frequent first:')
    for reason, n in reasons.most_common(45):
        print('  %7d  %s' % (n, reason))
    if stops:
        print('where the prototype stopped:')
        for stop, n in stops.most_common(30):
            print('  %7d  %s' % (n, stop))
    for note in kept[:30]:
        print(note[:400])
    bad = sum(n for (_, verdict), n in tally.items() if verdict.startswith(('DIFFERENT', 'ENDED')))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
