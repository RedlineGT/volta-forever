#!/usr/bin/env python3
"""ampere_check.py CUBIN|DIR...: the sm_80 words that pass to the translator as they are, against
NVIDIA's disassembler and against the two descriptions.

A word of sm_80 code passes unchanged when it matches a shared form (ampere-forms.txt) or a whole
class (ampere_forms.whole_classes): the claim is that it is the same instruction on sm_75. For
every distinct such word of the cubins (scheduler controls apart): nvdisasm must print the same
text for it as an instruction of sm_80 and of sm_75, and each description must find one class for
it, the sm_75 one being the sm_80 one's (its name, less the ending of a split class), with the same
operands. Prints the count and every word that fails; the exit status is 1 when one does.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ampere, ampere_forms, elf
from check import disassemble


def words_of(paths):
    for path in paths:
        files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith('.cubin')) if os.path.isdir(path) else [path]
        for name in files:
            e = elf.Elf(open(name, 'rb').read())
            if e.header[0x31 if e.header[8] >= 8 else 0x30] != 80:
                continue
            for s in e.sections:
                if s.name.startswith('.text.') and s.type == 1:
                    for at in range(0, len(s.data) - 15, 16):
                        yield int.from_bytes(s.data[at:at + 16], 'little')


def main():
    canonical, shared = ampere.supported()
    classes = ampere.whole()
    (d80, _), (d75, _) = ampere_forms.description(80), ampere_forms.description(75)
    seen, kept, how = set(), [], []
    for word in words_of(sys.argv[1:]):
        bits = word & ~ampere.CONTROLS
        if bits in seen:
            continue
        seen.add(bits)
        if any(ampere.matches(bits, form) for form in canonical.values()):
            continue
        if any(ampere.matches(bits, form) for form in shared):
            kept.append(word); how.append('form')
        elif any(ampere.matches(bits, form) for form in classes.get(bits & 0xfff, ())):
            kept.append(word); how.append('class')
    text80, text75 = [], []
    for i in range(0, len(kept), 4096):
        text80 += disassemble(kept[i:i + 4096], 80)[0]
        text75 += disassemble(kept[i:i + 4096], 75)[0]
    bad = 0
    for word, by, a, b in zip(kept, how, text80, text75):
        what = []
        if a is None or a != b:
            what.append('nvdisasm: sm_80 %r, sm_75 %r' % (a, b))
        try:
            c80 = [c for c in d80.candidates(word) if not c.alternate and d80.match(c, word)]
        except ValueError as error:
            c80 = []
            what.append('sm_80 description not read: %s' % str(error)[:60])
        c75 = d75.decode(word)
        if len(c80) != 1 or len(c75) != 1:
            what.append('classes: sm_80 %s, sm_75 %s' % ([c.name for c in c80], [c.name for c in c75]))
        else:
            name = c80[0].name
            if name != c75[0].name and not any(name.endswith(e) and name[:-len(e)] == c75[0].name for e in ampere_forms.SPLIT):
                what.append('class %s on sm_80, %s on sm_75' % (name, c75[0].name))
            o80 = {k: v for k, v in d80.operands(c80[0], word).items() if not k.startswith('TABLES_')}
            o75 = {k: v for k, v in d75.operands(c75[0], word).items() if not k.startswith('TABLES_')}
            if any(o75.get(k, v) != v for k, v in o80.items()):
                what.append('operands: sm_80 %s, sm_75 %s' % (o80, o75))
        if what:
            bad += 1
            if bad <= int(os.environ.get('SHOWN', 40)):
                print('%032x (%s): %s' % (word, by, '; '.join(what)))
    print('%d distinct sm_80 words (scheduler controls apart), %d pass unchanged (%d by a form, %d by a whole class): %d fail' % (
        len(seen), len(kept), how.count('form'), how.count('class'), bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main()
