#!/usr/bin/env python3
"""ampere_evidence.py CUBIN FUNCTION...: the forms of the sm_80 instructions of these functions that
pass to the translator by a whole class (ampere_forms.whole_classes), one a line, for
reference/ampere-evidence.txt.

A form is a class with the values of its modifiers: every operand that is an enumeration and not a
register, a predicate or a mark of the schedule, and every mark on an operand (negated, absolute,
inverted) but the inversion of a predicate. Registers, predicates and immediates are left free.
A whole class is the same instruction on sm_75 by the two descriptions; that its translation runs
right on a V100 is known only of the forms that ran there in translated sm_80 code and gave the
right results: those of the functions listed in ampere-evidence.txt, which this prints. FUNCTION
is a regular expression on the name; without one, every function of the cubin.
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ampere, ampere_forms, elf, md as M

FREE = M.REGISTER_TYPES + M.UNIFORM_TYPES + ('Predicate', 'UniformPredicate', 'BarrierRegister', 'BD', 'REQ', 'RD', 'WR',
                                             'USCHED_INFO', 'BATCH_T', 'REUSE')
SCHEDULE = ('req_bit_set', 'src_rel_sb', 'dst_wr_sb')


def modifiers(decoder, cls, word):
    """[(operand, value)] of the modifiers of an instruction of the class."""
    out = []
    for name, value in decoder.operands(cls, word).items():
        if name.startswith('TABLES_') or name in SCHEDULE:
            continue
        operand = cls.operands.get(name.split('@')[0])
        if '@' in name:
            if operand is None or operand.type not in ('Predicate', 'UniformPredicate'):
                out.append((name, value))
        elif operand is not None and operand.type in decoder.md.enums and operand.type not in FREE:
            out.append((name, value))
    return sorted(out)


def form_of(decoder, word):
    """The form of an sm_80 word as a line, or None when it is of no one class."""
    classes = [c for c in decoder.candidates(word) if not c.alternate]
    try:
        classes = [c for c in classes if decoder.match(c, word)]
    except ValueError:
        return None
    if len(classes) != 1:
        return None
    return ' '.join([classes[0].name] + ['%s=%d' % pair for pair in modifiers(decoder, classes[0], word)])


def main():
    decoder, _ = ampere_forms.description(80)
    canonical, shared = ampere.supported()
    # (Of every whole class, listed or not: this is how a form comes to be listed.)
    whole = {}
    for name, fixed, variable in ampere_forms.whole_classes()[0]:
        whole.setdefault(fixed & 0xfff, []).append((fixed, variable))
    wanted = [re.compile(r) for r in sys.argv[2:]]
    e = elf.Elf(open(sys.argv[1], 'rb').read())
    forms = set()
    for s in e.sections:
        if not (s.name.startswith('.text.') and s.type == 1):
            continue
        if wanted and not any(r.search(s.name[6:]) for r in wanted):
            continue
        for at in range(0, len(s.data) - 15, 16):
            word = int.from_bytes(s.data[at:at + 16], 'little')
            bits = word & ~ampere.CONTROLS
            if any(ampere.matches(bits, f) for f in canonical.values()) or any(ampere.matches(bits, f) for f in shared):
                continue
            if any(ampere.matches(bits, f) for f in whole.get(bits & 0xfff, ())):
                forms.add(form_of(decoder, word))
    for form in sorted(f for f in forms if f):
        print(form)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main()
