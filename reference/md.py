#!/usr/bin/env python3
"""Parser of NVIDIA's machine descriptions of the instruction sets, as extracted from nvdisasm by the
denvdis project (src/denvdis/data/sm70_1.txt, sm75_1.txt; fetched by scripts/fetch-all.sh, not in git).

A description gives, for each instruction class: its assembly format (operands and their types), the
opcode, and which bits of the 128-bit instruction word hold which operand. This module reads that
into Python objects and decodes instruction words with them. It is an offline tool: the translator
(crates/vf-sass) uses tables generated from it, and nvdisasm is the reference it is checked
against (check.py).

Bit numbering: bit 0 is the least significant bit of the first 64-bit word of an instruction.
"""
import os
import re
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src/denvdis/data')


class Field:
    """A named set of bits of the instruction word, most significant first."""
    __slots__ = ('name', 'bits', 'mask')

    def __init__(self, name, picture):
        assert len(picture) == 128, (name, len(picture))
        self.name = name
        self.bits = [127 - i for i, c in enumerate(picture) if c == 'X']
        self.mask = sum(1 << b for b in self.bits)

    def get(self, word):
        value = 0
        for b in self.bits:
            value = (value << 1) | ((word >> b) & 1)
        return value

    def put(self, word, value):
        for i, b in enumerate(reversed(self.bits)):
            word = (word & ~(1 << b)) | (((value >> i) & 1) << b)
        return word

    def __len__(self):
        return len(self.bits)


class Operand:
    """One operand or modifier of a class's format: `Type(args)*:name`."""
    __slots__ = ('type', 'args', 'star', 'name')

    def __init__(self, type_, args, star, name):
        self.type, self.args, self.star, self.name = type_, args, star, name

    def __repr__(self):
        return '%s%s%s:%s' % (self.type, '(%s)' % self.args if self.args is not None else '', '*' if self.star else '', self.name)


class Class:
    __slots__ = ('name', 'alternate', 'format', 'conditions', 'properties', 'predicates', 'opcodes', 'encoding',
                 'operands', 'opcode', 'opcode_field', 'mnemonic', 'line')

    def __repr__(self):
        return 'Class(%s)' % self.name


OPERAND = re.compile(r'([A-Za-z_][A-Za-z0-9_.]*)(?:\(([^()]*)\))?(\*)?:([A-Za-z_][A-Za-z0-9_.]*)')


class MD:
    def __init__(self, path):
        self.path = path
        text = open(path).read()
        first = re.search(r'^(?:ALTERNATE )?CLASS "', text, re.M).start()
        head, body = text[:first], text[first:]
        self.enums = {}      # name -> {symbol: value}
        self.defaults = {}   # name -> default symbol (marked '*=')
        self.tables = {}     # name -> [(keys, value)]
        self.fields = {}     # name -> Field
        self._head(head)
        self.classes = []
        self._classes(body, text[:first].count('\n'))
        self.by_name = {c.name: c for c in self.classes}

    # ---- header -------------------------------------------------------------------------------------
    def _head(self, head):
        registers = head.index('\nREGISTERS')
        tables = head.index('\nTABLES', registers)
        after = re.search(r'\nOPERATION PROPERTIES', head).start()
        funit = head.index('\nFUNIT uC')
        self._enums(head[registers + len('\nREGISTERS'):tables])
        self._tables(head[tables + len('\nTABLES'):after])
        for m in re.finditer(r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s+'([.X]{128})'", head[funit:], re.M):
            self.fields[m.group(1)] = Field(m.group(1), m.group(2))

    def _enums(self, text):
        for entry in text.split(';'):
            entry = entry.strip()
            if not entry:
                continue
            m = re.match(r'([A-Za-z_][A-Za-z0-9_.]*)\s*(.*)$', entry, re.S)
            name, rest = m.group(1), m.group(2)
            if rest.startswith('=') and '"' not in rest:
                # A union of enumerations defined before it.
                values = {}
                for part in rest[1:].split('+'):
                    values.update(self.enums[part.strip()])
                self.enums[name] = values
                continue
            values, nxt = {}, 0
            for item in self._split(rest):
                item = item.strip()
                if not item:
                    continue
                star = False
                m = re.match(r'^"?([^"=*(]+?)"?\s*(?:\(\s*(\d+)\s*\.\.\s*(\d+)\s*\))?\s*(\*)?\s*(?:=\s*(.+))?$', item, re.S)
                if not m:
                    raise ValueError('enum %s: %r' % (name, item))
                sym, lo, hi, star, value = m.group(1).strip(), m.group(2), m.group(3), m.group(4), m.group(5)
                if lo is not None:
                    lo, hi = int(lo), int(hi)
                    if value is not None:
                        vm = re.match(r'\(\s*(\d+)\s*\.\.\s*(\d+)\s*\)', value.strip())
                        base = int(vm.group(1))
                    else:
                        base = lo
                    for i in range(lo, hi + 1):
                        values['%s%d' % (sym, i)] = base + (i - lo)
                    nxt = base + (hi - lo) + 1
                else:
                    if value is not None:
                        nxt = int(value.strip(), 0)
                    values[sym] = nxt
                    if star:
                        self.defaults[name] = sym
                    nxt += 1
            self.enums[name] = values

    @staticmethod
    def _split(text):
        """Split on commas that are outside parentheses and quotes."""
        out, depth, cur, quote = [], 0, '', False
        for ch in text:
            if ch == '"':
                quote = not quote
            if not quote:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                elif ch == ',' and depth == 0:
                    out.append(cur)
                    cur = ''
                    continue
            cur += ch
        out.append(cur)
        return out

    def _tables(self, text):
        name = None
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line == ';':
                name = None
                continue
            if '->' in line:
                keys, value = line.rstrip(';').split('->')
                self.tables[name].append((tuple(keys.split()), value.strip()))
                if line.endswith(';'):
                    name = None
            else:
                name = line
                self.tables[name] = []

    # ---- classes ------------------------------------------------------------------------------------
    def _classes(self, body, line0):
        starts = [(m.start(), m.group(1) is not None, m.group(2)) for m in re.finditer(r'^(ALTERNATE )?CLASS "([^"]+)"', body, re.M)]
        for i, (pos, alternate, name) in enumerate(starts):
            end = starts[i + 1][0] if i + 1 < len(starts) else len(body)
            block = body[pos:end]
            c = Class()
            c.name, c.alternate, c.line = name, alternate, line0 + body[:pos].count('\n') + 1
            sections = {}
            marks = [(m.start(), m.group(1)) for m in re.finditer(r'^(FORMAT_ALIAS|FORMAT|CONDITIONS|PROPERTIES|PREDICATES|OPCODES|ENCODING|REMAP)\b', block, re.M)]
            for j, (p, key) in enumerate(marks):
                e = marks[j + 1][0] if j + 1 < len(marks) else len(block)
                sections[key] = block[p + len(key):e]
            c.format = sections.get('FORMAT', '').strip()
            c.conditions = sections.get('CONDITIONS', '').strip()
            c.properties = self._assignments(sections.get('PROPERTIES', ''))
            c.predicates = self._assignments(sections.get('PREDICATES', ''))
            c.opcodes = {}
            for m in re.finditer(r'([A-Za-z_0-9]+)\s*=\s*0b([01]+)\s*;', sections.get('OPCODES', '')):
                c.opcodes[m.group(1)] = int(m.group(2), 2)
            c.encoding = []
            for stmt in sections.get('ENCODING', '').split(';'):
                stmt = stmt.strip()
                if not stmt:
                    continue
                if stmt.startswith('!'):
                    c.encoding.append(([stmt[1:].strip()], '!'))
                    continue
                lhs, rhs = stmt.split('=', 1)
                c.encoding.append(([x.strip() for x in lhs.split(',')], rhs.strip()))
            c.operands = {}
            for m in OPERAND.finditer(c.format):
                c.operands[m.group(4)] = Operand(m.group(1), m.group(2), m.group(3) is not None, m.group(4))
            values = set(c.opcodes.values())
            assert len(values) == 1, (name, c.opcodes)
            c.opcode = values.pop()
            c.opcode_field = next(lhs[0] for lhs, rhs in c.encoding if rhs == 'Opcode')
            c.mnemonic = min(c.opcodes, key=len)
            self.classes.append(c)

    @staticmethod
    def _assignments(text):
        out = {}
        for stmt in text.split(';'):
            if '=' in stmt:
                k, v = stmt.split('=', 1)
                out[k.strip()] = v.strip()
        return out


def load(sm):
    return MD(os.path.join(DATA, 'sm%d_1.txt' % sm))


if __name__ == '__main__':
    for sm in (70, 75):
        md = load(sm)
        print('sm_%d: %d classes (%d alternate), %d enums, %d tables, %d fields' % (
            sm, len(md.classes), sum(c.alternate for c in md.classes), len(md.enums), len(md.tables), len(md.fields)))
        ofields = {}
        for c in md.classes:
            ofields.setdefault(c.opcode_field, 0)
            ofields[c.opcode_field] += 1
        print('  opcode fields:', ofields)
        types = {}
        for c in md.classes:
            for o in c.operands.values():
                types[o.type] = types.get(o.type, 0) + 1
        print('  %d operand types; not enumerations:' % len(types), sorted(t for t in types if t not in md.enums))


# ---- decoding ---------------------------------------------------------------------------------------

REGISTER_TYPES = ('Register', 'NonZeroRegister', 'ZeroRegister', 'RegisterFAU', 'NonZeroRegisterFAU', 'ZeroRegisterFAU')
UNIFORM_TYPES = ('UniformRegister', 'NonZeroUniformRegister', 'ZeroUniformRegister')


class Decoder:
    """Finds the class of an instruction word and the values of its operands."""

    def __init__(self, md):
        self.md = md
        self.by_opcode = {}
        for c in md.classes:
            f = md.fields[c.opcode_field]
            self.by_opcode.setdefault((c.opcode_field, c.opcode), []).append(c)
        self.opcode_fields = sorted({c.opcode_field for c in md.classes}, key=lambda n: -len(md.fields[n]))
        self._plans = {}
        # Of this description only: six enumerations of the same name have other values in sm_70 than
        # in sm_75 (F16ONLY, F32ONLY, HIONLY, LODLC, ONLY64, U32ONLY).
        self._domains = {}

    def plan(self, c):
        """The encoding of a class as a list of (kind, fields, data) steps."""
        p = self._plans.get(c.name)
        if p is not None:
            return p
        md, p = self.md, []
        for lhs, rhs in c.encoding:
            fields = [md.fields[n] for n in lhs]
            if rhs == '!':
                p.append(('unused', fields, None))
            elif rhs == 'Opcode':
                p.append(('opcode', fields, c.opcode))
            elif re.fullmatch(r'\*\s*\d+', rhs):
                p.append(('default', fields, int(rhs[1:])))
            elif re.fullmatch(r'\d+', rhs):
                p.append(('const', fields, int(rhs)))
            elif re.fullmatch(r'\*?\s*[A-Za-z_][A-Za-z0-9_.]*', rhs):
                p.append(('operand', fields, rhs.lstrip('* ')))
            elif re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*@[a-z]+', rhs):
                p.append(('attribute', fields, tuple(rhs.split('@'))))
            elif rhs.startswith('TABLES_'):
                name, args = rhs[:-1].split('(')
                p.append(('table', fields, (name, [a.strip() for a in args.split(',')])))
            elif ' SCALE ' in rhs:
                name, scale = rhs.split(' SCALE ')
                p.append(('scaled', fields, (name.strip(), int(scale))))
            elif re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]* convertFloatType\(.*\)', rhs):
                # (sm_80: an immediate whose bits are read as one type of float or another, by a
                # modifier. The bits are the operand's either way.)
                p.append(('converted', fields, rhs.split()[0]))
            elif re.fullmatch(r'[A-Za-z_0-9]+\(.*\)', rhs):
                name, args = rhs[:-1].split('(', 1)
                p.append(('function', fields, (name, [a.strip() for a in args.split(',')])))
            else:
                raise ValueError('%s: encoding %r' % (c.name, rhs))
        self._plans[c.name] = p
        return p

    def candidates(self, word):
        out = []
        for name in self.opcode_fields:
            out += self.by_opcode.get((name, self.md.fields[name].get(word)), [])
        return out

    def fixed(self, c, word, strict_defaults):
        """Whether the constant fields of class `c` have their values in `word`."""
        for kind, fields, data in self.plan(c):
            if kind == 'const' or (kind == 'default' and strict_defaults):
                if fields[0].get(word) != data:
                    return False
        return True

    def operands(self, c, word):
        """The raw field values of the operands of class `c` in `word`: {operand or operand@attribute: value}."""
        out = {}
        for kind, fields, data in self.plan(c):
            if kind in ('operand', 'converted'):
                out[data] = fields[0].get(word)
            elif kind == 'attribute':
                out['%s@%s' % data] = fields[0].get(word)
            elif kind == 'scaled':
                out[data[0]] = fields[0].get(word) * data[1]
            elif kind == 'table':
                out['%s(%s)' % (data[0], ','.join(data[1]))] = fields[0].get(word)
            elif kind == 'function':
                for f, a in zip(fields, data[1]):
                    out[a] = f.get(word)
        return out

    # -- full match --------------------------------------------------------------------------------------
    def _domain(self, c, name):
        """The set of raw values that operand `name` of class `c` may have, or None when any value will do."""
        o = c.operands.get(name)
        if o is None:
            return None
        values = self.md.enums.get(o.type)
        if values is None:
            return None
        key = o.type
        d = self._domains.get(key)
        if d is None:
            d = self._domains[key] = frozenset(v for s, v in values.items() if not s.startswith('INVALID'))
        return d

    def match(self, c, word):
        """Whether `word` is an instruction of class `c`: constants, defaults, and operands within their types."""
        for kind, fields, data in self.plan(c):
            if kind in ('const', 'default'):
                if fields[0].get(word) != data:
                    return False
            elif kind == 'operand':
                d = self._domain(c, data)
                if d is not None and fields[0].get(word) not in d:
                    return False
            elif kind == 'table':
                values = self._table_values(data[0])
                if fields[0].get(word) not in values:
                    return False
        return True

    _tables = {}

    def _table_values(self, name):
        key = (id(self.md), name)
        v = self._tables.get(key)
        if v is None:
            v = self._tables[key] = frozenset(int(value, 0) for keys, value in self.md.tables[name] if re.fullmatch(r'-?\d+|0x[0-9a-fA-F]+', value))
        return v

    def decode(self, word, alternates=False):
        """The classes that `word` is an instruction of."""
        return [c for c in self.candidates(word) if (alternates or not c.alternate) and self.match(c, word)]

    def unused_clear(self, c, word):
        """Whether the bits that class `c` does not use are zero in `word`."""
        for kind, fields, data in self.plan(c):
            if kind == 'unused' and any(f.get(word) for f in fields):
                return False
        return True
