#!/usr/bin/env python3
"""What the translator knows of the two instruction sets: classes, what instructions read and write,
the kind each is for scheduling, the rules for uniform operands, the distances native code keeps.
"""
import collections, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import md as M
import remap
from latency import operation_sets, kind_of, Uses, registers

HERE = os.path.dirname(os.path.abspath(__file__))
ZERO = {'R': 255, 'U': 63, 'P': 7, 'Q': 7}
USCHED, WRITE_BARRIER, WAIT = 105, 110, 116        # first bits of the stall field, of the scoreboard written, of the wait mask
READ_BARRIER = 113                                # of the scoreboard on which an instruction says it has read its operands
REUSE = 7 << 122                                  # with the batch field: the three bits above the wait mask
NOP = 0x000fc000000000000000000000007918
# The kinds of instruction that read their operands some time after they are issued: memory accesses
# and conversions, arithmetic on doubles, matrix products. They are the kinds that name a scoreboard
# for having read, in native code (a fifth of PyTorch's and cuBLAS's sm_70 code: 200,000 instructions
# of these kinds do, none of another kind).
LATE = ('MIO_FAST', 'MIO_SLOW', 'FMALITE', 'HMMA')
# Memory instructions of sm_75 whose address has a uniform part, by the start of the name of their
# class, and the sm_70 class of the plain instruction (its address in a register).
UNIFORM_ADDRESS = {'ldg_uniform__': 'ldg__sImmOffset', 'stg_uniform__': 'stg__sImmOffset',
                   'ld_uniform__': 'ld__sImmOffset', 'st_uniform__': 'st__sImmOffset',
                   'atomg_uniform__': 'atomg__RaNonRZ', 'red_uniform__': 'red__RaNonRZ',
                   'sts_uniform_': 'sts__sImmOffset', 'lds_uniform_': 'lds__sImmOffset'}
# Of those plain classes, the ones in shared memory: an address of 32 bits, whose register may have
# a stride.
SHARED = ('sts__sImmOffset', 'lds__sImmOffset')
# The classes whose address register may have a stride on sm_75 and has none on sm_70 (generate.py
# checks that they are those the descriptions say).
STRIDED = ('sts__sImmOffset', 'lds__sImmOffset', 'lds_zd__sImmOffset',
           'atoms__RaNonRZ', 'atoms_cas__RaNonRZ_CAS', 'atoms_cas__RaNonRZ_CAST')


# The instructions whose results Volta's code reads no earlier than a fixed distance, also behind
# a scoreboard (`Machine.waited`).
WAITED = ('DADD', 'DMUL', 'DFMA', 'DSETP')


class Decoded:
    """An instruction word with its class and what it reads and writes."""
    __slots__ = ('word', 'c', 'ops', 'kind', 'reads', 'writes', 'mnemonic')


class Machine:
    def __init__(self, distances=None):
        self.md75, self.md70 = M.load(75), M.load(70)
        self.d75, self.d70 = M.Decoder(self.md75), M.Decoder(self.md70)
        self.sets75, self.sets70 = operation_sets(75), operation_sets(70)
        self.rules = {}
        for name, (good, bad) in remap.candidates(self.md75, self.d75, self.md70, self.d70).items():
            if good:
                self.rules[name] = good
        # The rules that nvdisasm confirmed (remap.py): sm_75 class -> sm_70 class.
        self.rule_of = {}
        for line in open(os.path.join(HERE, 'rules-sm75.tsv')):
            if not line.startswith('#'):
                a, b, n = line.rstrip('\n').split('\t')
                self.rule_of[a] = b
        self._cache = {75: {}, 70: {}}
        self._rules = {}
        self._uses = {}
        # The distances of distances.py: by kind of writer, and by mnemonic ('=NAME') where longer.
        self.distance = {}
        path = distances or os.path.join(HERE, 'distances-sm70.tsv')
        for line in open(path):
            f = line.rstrip('\n').split('\t')
            if len(f) >= 5 and not line.startswith('#'):
                self.distance[(f[0], f[1], f[2], f[3])] = int(f[4])

    def rule(self, a, b, drop=()):
        """The rule from sm_75 class `a` to sm_70 class `b` (names), or None when their fields do not
        correspond. `drop`: fields of `a` left to the caller."""
        key = (a, b, drop)
        if key not in self._rules:
            r = remap.Rule(self.md75, self.d75, self.md75.by_name[a], self.md70, self.d70, self.md70.by_name[b], drop) if b in self.md70.by_name else None
            self._rules[key] = r if r is not None and r.ok else None
        return self._rules[key]

    def decode(self, sm, word):
        """The instruction `word` is on architecture `sm`, or None when it is not exactly one."""
        cache = self._cache[sm]
        if word in cache:
            return cache[word]
        md, dec, sets = (self.md75, self.d75, self.sets75) if sm == 75 else (self.md70, self.d70, self.sets70)
        cs = dec.decode(word)
        if sm == 70:
            # What is given to Volta has no bit set that its class does not use: native code has none.
            cs = [c for c in cs if dec.unused_clear(c, word)]
        r = None
        if len(cs) == 1:
            c = cs[0]
            uses = self._uses.get((sm, c.name))
            if uses is None:
                uses = self._uses[(sm, c.name)] = Uses(md, c)
            r = Decoded()
            r.word, r.c, r.ops, r.mnemonic = word, c, dec.operands(c, word), c.mnemonic
            r.kind = kind_of(sets, c.mnemonic)
            r.reads = [(l, r.ops[n] + i, 'guard' if n in ('Pg', 'UPg') else 'data') for l, n, s in uses.reads if n in r.ops and r.ops[n] != ZERO[l] for i in range(registers(s, r.ops))]
            r.writes = [(l, r.ops[n] + i) for l, n, s in uses.writes if n in r.ops and r.ops[n] != ZERO[l] for i in range(registers(s, r.ops))]
        cache[word] = r
        return r

    def least(self, letter, role, writer, reader_kind):
        """The distance native sm_70 code keeps from an instruction `writer` (a Decoded) that writes a
        register (letter R) or a predicate (P) to a reader of kind `reader_kind`."""
        d = self.distance.get((letter, role, '=' + writer.mnemonic, reader_kind))
        if d is None:
            d = self.distance.get((letter, role, writer.kind, reader_kind))
        if d is None:
            # Never seen together: the longest distance this kind of writer keeps from anything, or 15.
            d = max([v for (l, r, w, k), v in self.distance.items() if l == letter and w == writer.kind] or [15])
        if letter == 'P' and writer.mnemonic == 'R2P':
            # Predicates given back from a register (the tables do not say which R2P writes: its
            # mask is a number). In native sm_70 code a plain instruction reads one as an operand
            # 5 cycles later or more (4,346 of them in half of PyTorch and cuBLAS); the others seen
            # (guards of memory instructions) 12 or more: 12 for everything else.
            d = max(d, 5 if (reader_kind, role) == ('FXU', 'data') else 12)
        return d

    def waited(self, letter, role, writer, reader_kind):
        """The distance that native sm_70 code keeps from `writer` to a reader that waits for the
        scoreboard on which it announces its result, where a wait is not all it keeps: None for
        every writer but the four of double precision. A reader of what DADD, DMUL, DFMA or
        DSETP writes never comes closer than this in Volta's own code, scoreboard or not (8
        cycles to an instruction of the same pipe, 9 to any other; a predicate of DSETP 15 as a
        guard or to a branch), where code for sm_80 has 6 and 14 (docs/validation.md, section 8:
        measured on native code, and what the compiler's own tables say)."""
        if writer.mnemonic not in WAITED:
            return None
        if letter == 'P' and (role == 'guard' or reader_kind == 'BRU'):
            return 15
        return 8 if reader_kind == 'FMALITE' else 9

    def longest(self, letter, writer):
        """The distance that is enough from `writer` to any reader of what it writes."""
        return max(self.least(letter, role, writer, kind) for role in (('data', 'guard') if letter == 'P' else ('data',))
                   for kind in ('FXU', 'IMAD', 'FMAI', 'FMALITE', 'MIO_FAST', 'MIO_SLOW', 'BRU', 'HMMA'))

    # ---- building sm_70 instructions -----------------------------------------------------------------
    def assemble(self, name, **given):
        """The word of an instruction of sm_70 class `name`: operands by name (enumerations by symbol
        or value), what is not given at its default, no scoreboard, a stall of one."""
        md, dec = self.md70, self.d70
        c = md.by_name[name]
        word, used = 0, set()
        def value(operand):
            if operand in given:
                used.add(operand)
                v = given[operand]
                if isinstance(v, str):
                    v = md.enums[c.operands[operand].type][v]
                return v
            return remap.default_raw(md, c, operand)
        for kind, fields, data in dec.plan(c):
            if kind in ('opcode', 'const', 'default'):
                word = fields[0].put(word, data)
            elif kind == 'operand':
                word = fields[0].put(word, value(data))
            elif kind == 'attribute':
                key = '%s_%s' % data
                used.add(key)
                word = fields[0].put(word, given.get(key, 0))
            elif kind == 'scaled':
                v = value(data[0])
                assert v % data[1] == 0
                word = fields[0].put(word, v // data[1])
            elif kind == 'function':
                for f, a in zip(fields, data[1]):
                    word = f.put(word, value(a))
            elif kind == 'table' and fields[0].name == 'BITS_8_124_122_109_105_opex':
                word |= 1 << USCHED
        assert used >= set(k for k in given), (name, set(given) - used)
        assert dec.match(c, word) and dec.unused_clear(c, word), (name, hex(word))
        return word


def stall_of(word):
    return (word >> USCHED) & 15


def with_stall(word, stall, marks=False):
    """`word` with a stall of `stall` cycles (1 to 15), keeping how it ends its group when it can.

    The three bits above the wait mask go with the stall. In an instruction whose operands can
    bear a mark of register reuse (`marks`), they are those marks when the stall ends a group (17
    to 27 in the field) and name a batch with any other (NVIDIA's tables `TABLES_opex_*`, the
    same in all of them). So an instruction with marks that can no longer end its group loses
    them: they are hints, and BATCH_START in their place is not one. And one that names a batch
    is not made to end its group. (Until 2026-11-17 the marks were left where they were: found
    on code the translator had never seen.)"""
    assert 0 <= stall <= 15
    old = (word >> USCHED) & 31
    keeps = old >> 4 and 1 <= stall <= 11
    if marks and word & REUSE:
        if 17 <= old <= 27:
            if not keeps:
                word &= ~REUSE
        else:
            keeps = False
    return (word & ~(31 << USCHED)) | ((stall | (16 if keeps else 0)) << USCHED)


def marked(d):
    """The decoded instruction `d` has operands that can bear a mark of register reuse."""
    return d is not None and any('reuse_src' in key for key in d.ops)
