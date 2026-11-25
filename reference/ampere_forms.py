#!/usr/bin/env python3
"""Generate the audited sm80 encodings used by vf-sass and its Python reference."""
from functools import lru_cache
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import md

HERE = Path(__file__).resolve().parent
CONTROL = ((1 << 17) - 1) << 105
# These forms have different encodings on Ampere and Turing.
CANONICAL = {
    # A global load or store through a memory descriptor, of any size and guard.
    "LOAD": "ldg_memdesc__Ra64 Pg Pg.not Rd Ra Ra_URb Ra_offset sz | e=E cop=EN Pu=PT Pnz=PT input_reg_sz_64_dist=64 sem=WEAK",
    "LOAD_CONST": "ldg_memdesc__Ra64 Pg Pg.not Rd Ra Ra_URb Ra_offset sz | e=E cop=EN Pu=PT Pnz=PT input_reg_sz_64_dist=64 sem=CONSTANT",
    "STORE": "stg_memdesc__Ra64 Pg Pg.not Ra Rb Ra_URc Ra_offset sz | e=E cop=EN input_reg_sz_64_dist=64 sem=WEAK",
    "BRA_MASK_ALL": "bra_uniform_ sImm | Pg=PT Pp=PT cond=DIV URb=URZ URb.invert=1",
    "ATOM_ADD32": "atomg_uniform__memdesc Rd Ra Ra_URc Rb Ra_offset | Pg=PT e=E op=ADD cop=EN sz=32 Pu=PT input_reg_sz_64_dist=64 sem=STRONG sco=GPU",
    "SHARED_INC": "atoms_arrive__popcinc Pg Pg.not Ra Ra_offset | Rd=RZ stride=X4 Ra_URc=URZ op=POPC.INC sz=32",
    "COPY128": "ldgsts__desc_RRU Pg Pg.not Rb Rb_offset Ra_URc Ra Ra_offset | loc=ACCESS cop=EN sz=128 input_reg_sz_64_dist=64 Pnz=PT sem=WEAK",
    "COMMIT": "ldgdepbar_ | Pg=PT",
    "HALF_CONSTANT": "hfma2_mma__RRI_norelu Rd Sc Sc1 | Pg=PT Ra=RZ Ra.negate=1 Rb=RZ",
    "BF16_ADD": "hfma2__RIR Rd Ra Rc | Pg=PT ofmt=BF16_V2 fmz=nofmz satrelu=nosat iswzA=H1_H0 iswzC=H1_H0 Sb=0x3f80 Sb1=0x3f80",
    "BF16_PACK": "f2fp__RRR Rd Ra Rb | Pg=PT dstfmt=BF16 merge=PACK_AB rndMode=RN",
    "PACK": "f2fp__RRR Rd Ra Rb | Pg=PT dstfmt=F16 merge=PACK_AB rndMode=RN",
}


@lru_cache(maxsize=2)
def description(sm):
    data = Path(md.DATA)
    path = data if sm == 75 else data.parent / "data12"
    machine = md.MD(str(path / f"sm{sm}_1.txt"))
    return md.Decoder(machine), {c.name: c for c in machine.classes}


def encode(specification, sm):
    head, fixed = specification.split("|")
    names, *variable = head.split()
    cls_name = names.split("/")[0 if sm == 80 else -1]
    decoder, classes = description(sm)
    cls = classes[cls_name]
    # Some split Ampere classes make a Turing operand implicit, e.g. ISETP without EX.
    pins = {}
    for item in fixed.split():
        name, spelling = item.split("=")
        if name.startswith(("sm75.", "sm80.")):
            target, name = name.split(".", 1)
            if target != f"sm{sm}":
                continue
        pins[name] = spelling
    seen = set()
    positions = {}

    def value(name):
        seen.add(name)
        spelling = pins.get(name)
        if spelling is None:
            return 0
        operand = cls.operands.get(name)
        if operand:
            values = decoder.md.enums.get(operand.type, {})
            if spelling in values:
                return values[spelling]
        return int(spelling, 0)

    word = mask = 0
    for kind, fields, data in decoder.plan(cls):
        if kind == "unused":
            continue
        if all((f.mask & ~CONTROL) == 0 for f in fields):
            continue
        if kind in ("opcode", "const", "default"):
            word = fields[0].put(word, data)
            continue
        if kind == "table":
            table, args = data
            seen.update(args)
            # The scoreboard and issue interval are scheduled separately. A legal non-batch
            # issue interval allows the requested operand reuse markers to be encoded.
            if table.startswith("TABLES_opex_"):
                wanted = [value(a) if a != "usched_info" else None for a in args]
            else:
                wanted = [value(a) for a in args]
            choices = [int(encoded, 0) for keys, encoded in decoder.md.tables[table]
                       if all(v is None or int(k, 0) == v for v, k in zip(wanted, keys, strict=True))]
            high = {fields[0].put(0, v) & ~CONTROL for v in choices}
            if len(high) != 1:
                raise ValueError(f"{cls.name} has no unique encoding for {args}={wanted}: {high}")
            word |= high.pop()
            continue
        if kind in ("operand", "converted"):
            args = [data]
        elif kind == "attribute":
            args = [".".join(data)]
        elif kind == "scaled":
            args = [data[0]]
        elif kind == "function" and data[0] in ("ConstBankAddress0", "ConstBankAddress2"):
            args = data[1]
        else:
            raise ValueError(f"unhandled {cls.name} encoding {kind} {data}")
        for name, field in zip(args, fields, strict=True):
            if name in variable:
                seen.add(name)
                positions[name] = tuple(field.bits)
                mask |= field.mask
            else:
                word = field.put(word, value(name))
    unknown = (set(variable) | pins.keys()) - seen
    if unknown or set(variable) & pins.keys():
        raise ValueError(f"{cls.name} has invalid operands: {unknown or set(variable) & pins.keys()}")
    return word & ~CONTROL, mask, positions


@lru_cache(maxsize=1)
def shared():
    records = []
    for line in (HERE / "ampere-forms.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        ampere = encode(line, 80)
        turing = encode(line, 75)
        if ampere != turing:
            raise ValueError(f"sm80 and sm75 encode different instructions: {line}\n{ampere}\n{turing}")
        records.append((line.split()[0], *ampere[:2]))
    return records


def forms():
    return [(fixed, variable) for _, fixed, variable in shared()]


def canonical_forms():
    masked = encode(CANONICAL["BRA_MASK_ALL"], 80)
    native = encode("bra_ sImm | Pg=PT Pp=PT cond=DIV", 75)
    if masked[2]["sImm"] != native[2]["sImm"]:
        raise ValueError("BRA.DIV displacement occupies different bits on sm80 and sm75")
    return {name: encode(spec, 80)[:2] for name, spec in CANONICAL.items()}


# ---- whole classes ----------------------------------------------------------------------------------
# A class of sm80 whose every bit holds what the same bit of the sm75 class of the same name holds
# is the same instruction on both: its words pass as they are. The comparison is of the two
# descriptions, bit by bit, and nothing of a class is taken on trust beyond its name.

# The sm80 classes that one sm75 class was split into, by the ending of their names.
SPLIT = ("_noEX", "_EX")
# More forms than this for one class (an operand whose values differ, taken value by value): left out.
MOST_FORMS = 512


class Unread(ValueError):
    pass


def _enum(md, cls, name):
    """(type, {symbol: value} or None, arguments) of an operand of a class."""
    operand = cls.operands.get(name.split("@")[0])
    if operand is None:
        raise Unread(f"{cls.name}: no operand {name}")
    return operand.type, md.enums.get(operand.type), operand.args


def _default(md, cls, name):
    """The value that an operand has when the text of the instruction leaves it out."""
    if "@" in name:
        return 0
    kind, values, args = _enum(md, cls, name)
    if values is None or args is None:
        return None
    spelling = args.strip('"')
    return values.get(spelling, values.get("_" + spelling))


def layout(sm, cls):
    """What each bit of an instruction of the class holds: {bit: meaning}. A meaning is
    ('fixed', bit value), or (kind, operand or operands, significance within the field, ...)."""
    decoder, _ = description(sm)
    md, bits = decoder.md, {}

    def put(field, meaning):
        for i, b in enumerate(reversed(field.bits)):
            if b in bits:
                raise Unread(f"{cls.name}: bit {b} twice")
            bits[b] = meaning(i)

    try:
        plan = decoder.plan(cls)
    except ValueError as error:
        raise Unread(str(error)) from None
    for kind, fields, data in plan:
        if kind == "unused":
            for f in fields:
                put(f, lambda i: ("fixed", 0))
        elif kind in ("opcode", "const", "default"):
            put(fields[0], lambda i: ("fixed", data >> i & 1))
        elif kind == "operand":
            put(fields[0], lambda i: ("operand", data, i))
        elif kind == "attribute":
            put(fields[0], lambda i: ("operand", "%s@%s" % data, i))
        elif kind == "scaled":
            put(fields[0], lambda i: ("scaled", data[0], i, data[1]))
        elif kind in ("table", "function"):
            name, args = data
            if kind == "function" and len(fields) != 1:
                # One field for each argument (the bank and the offset of a constant).
                for f, a in zip(fields, args, strict=True):
                    put(f, lambda i, a=a: ("function", (a,), i, name))
            else:
                put(fields[0], lambda i: (kind, tuple(args), i, name))
        else:
            raise Unread(f"{cls.name}: encoding {kind} {data}")
    return bits


def _values_agree(md80, cls80, md75, cls75, name):
    """For an operand in both classes: None when every value reads the same on both, else the
    values that do. Raises Unread when the operand is not of the same kind on both."""
    type80, values80, args80 = _enum(md80, cls80, name)
    type75, values75, args75 = _enum(md75, cls75, name)
    if (values80 is None) != (values75 is None):
        raise Unread(f"{cls80.name}: {name} is {type80} and {type75}")
    if values80 is None:
        # An immediate: the same type, width and sign.
        if type80 != type75 or args80.split("/")[0] != args75.split("/")[0]:
            raise Unread(f"{cls80.name}: {name} is {type80}({args80}) and {type75}({args75})")
        return None
    names80, names75 = {}, {}
    for symbol, value in values80.items():
        names80.setdefault(value, set()).add(symbol)
    for symbol, value in values75.items():
        names75.setdefault(value, set()).add(symbol)
    same = {v for v, names in names80.items()
            if names <= names75.get(v, set()) and not any(n.startswith("INVALID") for n in names)}
    valid = {v for v, names in names80.items() if not all(n.startswith("INVALID") for n in names)}
    return None if same >= valid else same


def _table_within(md80, md75, name80, name75):
    """Every row of the sm80 table is a row of the sm75 table: a value that sm80 gives a field
    stands for the same operands on sm75."""
    if name80 not in md80.tables or name75 not in md75.tables:
        return name80 == name75 and name80 not in md80.tables and name75 not in md75.tables
    return set(md80.tables[name80]) <= set(md75.tables[name75])


def _class_forms(cls80, cls75):
    """The forms (fixed bits, variable bits) of the words of an sm80 class that are the same
    instruction of the sm75 class, or Unread."""
    (d80, _), (d75, _) = description(80), description(75)
    md80, md75 = d80.md, d75.md
    a, b = layout(80, cls80), layout(75, cls75)
    if cls80.mnemonic != cls75.mnemonic or cls80.properties.get("INSTRUCTION_TYPE") != cls75.properties.get("INSTRUCTION_TYPE"):
        raise Unread(f"{cls80.name}: another mnemonic or type of instruction")
    if a.keys() != b.keys():
        raise Unread(f"{cls80.name}: not the same bits")
    fixed = variable = 0
    only80, only75 = {}, {}        # operand: {significance: (bit, value the other side fixes)}
    both = {}                      # operand: [bit by significance]
    for bit in sorted(a):
        x, y = a[bit], b[bit]
        if x[0] == "fixed" and y[0] == "fixed":
            if x != y:
                raise Unread(f"{cls80.name}: bit {bit} is {x[1]} and {y[1]}")
            fixed |= x[1] << bit
        elif x[0] == "fixed":
            if y[0] != "operand":
                raise Unread(f"{cls80.name}: bit {bit} is fixed, and of {y[0]} {y[1]} on sm75")
            only75.setdefault(y[1], {})[y[2]] = (bit, x[1])
            fixed |= x[1] << bit
        elif y[0] == "fixed":
            if x[0] != "operand":
                raise Unread(f"{cls80.name}: bit {bit} is of {x[0]} {x[1]}, and fixed on sm75")
            only80.setdefault(x[1], {})[x[2]] = (bit, y[1])
            fixed |= y[1] << bit
        else:
            if x[:3] != y[:3]:
                raise Unread(f"{cls80.name}: bit {bit} is of {x[1]} and of {y[1]}")
            if x[0] == "scaled" and x[3] != y[3]:
                raise Unread(f"{cls80.name}: {x[1]} is scaled by {x[3]} and by {y[3]}")
            if x[0] in ("table", "function") and not _table_within(md80, md75, x[3], y[3]):
                raise Unread(f"{cls80.name}: {x[3]} and {y[3]} differ")
            if x[0] in ("table", "function") and x[3] != y[3] and not x[3].startswith("TABLES_"):
                raise Unread(f"{cls80.name}: {x[3]} and {y[3]}")
            variable |= 1 << bit
            names = x[1] if isinstance(x[1], tuple) else (x[1],)
            for name in names:
                both.setdefault(name, {})
            if x[0] == "operand":
                both[x[1]][x[2]] = bit
    # An operand that sm75 has in more bits than sm80, or that sm80 has not: the value that the
    # bits fixed by sm80 give it.
    for name, places in only75.items():
        if name in both:
            if any(value for _, value in places.values()) or min(places) < len(both[name]):
                raise Unread(f"{cls80.name}: {name} is wider on sm75, and not by zeros above")
        else:
            cover = {m[2] for m in b.values() if m[0] == "operand" and m[1] == name}
            value = sum(v << i for i, (_, v) in places.items())
            if cover != set(places) or _default(md75, cls75, name) != value:
                raise Unread(f"{cls80.name}: {name}, which sm80 has not, is not left at its default")
    choices = []
    for name, places in only80.items():
        if name in both:
            if any(value for _, value in places.values()) or min(places) < len(both[name]):
                raise Unread(f"{cls80.name}: {name} is wider on sm80, and not by zeros above")
        else:
            cover = {m[2] for m in a.values() if m[0] == "operand" and m[1] == name}
            value = sum(v << i for i, (_, v) in places.items())
            if cover != set(places) or _default(md80, cls80, name) != value:
                raise Unread(f"{cls80.name}: {name}, which sm75 has not, has not its default there")
    for name in sorted(both):
        if "@" in name:
            if name.split("@")[0] not in cls75.operands:
                raise Unread(f"{cls80.name}: no operand {name} on sm75")
            continue
        agree = _values_agree(md80, cls80, md75, cls75, name)
        if agree is None:
            continue
        places = both[name]
        if not places or name in only75 or name in only80:
            # In a table, or of two widths: only when every value reads the same.
            raise Unread(f"{cls80.name}: values of {name} differ")
        mask = sum(1 << bit for bit in places.values())
        choices.append((mask, [sum((value >> i & 1) << bit for i, bit in places.items())
                               for value in sorted(agree) if value < 1 << len(places)]))
    forms = [(fixed, variable)]
    for mask, values in choices:
        forms = [(f | value, v & ~mask) for f, v in forms for value in values]
        if len(forms) > MOST_FORMS:
            raise Unread(f"{cls80.name}: too many forms")
    return forms


def _pattern(sm, cls):
    """The fixed bits of a class and their mask, of its opcode alone when the rest is not read."""
    decoder, _ = description(sm)
    try:
        bits = layout(sm, cls)
    except Unread:
        field = decoder.md.fields[cls.opcode_field]
        return field.put(0, cls.opcode), field.mask
    return (sum(1 << b for b, m in bits.items() if m == ("fixed", 1)),
            sum(1 << b for b, m in bits.items() if m[0] == "fixed"))


def _class_pattern(cls):
    """(fixed, variable) of an sm80 class with nothing of it held back."""
    fixed, mask = _pattern(80, cls)
    return fixed, ~mask & (1 << 128) - 1


@lru_cache(maxsize=1)
def whole_classes():
    """[(sm80 class, fixed, variable)], sorted, and {sm80 class: why it is not there}."""
    (_, c80), (_, c75) = description(80), description(75)
    found, why, partner = {}, {}, {}
    for name, cls in c80.items():
        if cls.alternate:
            continue
        target = name
        if name not in c75:
            target = next((name[:-len(e)] for e in SPLIT if name.endswith(e) and name[:-len(e)] in c75), None)
        if target is None or c75[target].alternate:
            why[name] = "no such class on sm75"
            continue
        try:
            found[name] = _class_forms(cls, c75[target])
            partner[name] = target
        except Unread as error:
            why[name] = str(error)
    # A word of a class kept here must be of no class, on either side, that is not kept whole:
    # the decoders of both then name the same instruction. (A class kept for some values of an
    # operand only is not kept whole: a word with another value may be of a class beside it.)
    whole = {name for name, forms in found.items() if len(forms) == 1 and forms[0] == _class_pattern(c80[name])}
    whole75 = {partner[name] for name in whole}
    def clashes(sm, classes, name, kept):
        fixed, mask = _pattern(sm, classes[name])
        return [o for o, other in classes.items() if o != name and o not in kept and not other.alternate
                and other.opcode == classes[name].opcode
                and (fixed ^ _pattern(sm, other)[0]) & mask & _pattern(sm, other)[1] == 0]
    result = []
    for name in sorted(found):
        clash = clashes(80, c80, name, whole) + clashes(75, c75, partner[name], whole75)
        if clash:
            why[name] = "its words may be of " + ", ".join(clash[:3])
            continue
        for fixed, variable in found[name]:
            assert not fixed & variable
            result.append((name, fixed & ~CONTROL, variable & ~CONTROL))
    return result, why


@lru_cache(maxsize=1)
def evidenced_forms():
    """[(sm80 class, fixed, variable)]: the forms of ampere-evidence.txt, each a whole class
    (`whole_classes`) with its modifiers at the values listed. Only these pass by their class."""
    (decoder, classes), whole = description(80), {}
    for name, fixed, variable in whole_classes()[0]:
        whole.setdefault(name, []).append((fixed, variable))
    result = set()
    for line in (HERE / "ampere-evidence.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, *pins = line.split()
        if name not in whole:
            raise ValueError(f"ampere-evidence.txt: {name} is not a whole class")
        places = {}
        for kind, fields, data in decoder.plan(classes[name]):
            if kind == "operand":
                places[data] = fields[0]
            elif kind == "attribute":
                places["%s@%s" % data] = fields[0]
        pinned = pin_mask = 0
        for pin in pins:
            operand, value = pin.split("=")
            if operand not in places:
                raise ValueError(f"ampere-evidence.txt: {name} has no {operand}")
            pinned = places[operand].put(pinned, int(value))
            pin_mask |= places[operand].mask
        # Every modifier of the class is named: a form says all of them.
        import ampere_evidence
        named = {operand for operand, _ in ampere_evidence.modifiers(decoder, classes[name], 0)}
        if named != {pin.split("=")[0] for pin in pins}:
            raise ValueError(f"ampere-evidence.txt: {line}: the modifiers of {name} are {sorted(named)}")
        for fixed, variable in whole[name]:
            # (A class kept for some values of an operand only: the forms that agree with the pins.)
            if (fixed ^ pinned) & pin_mask & ~variable & ~CONTROL:
                continue
            result.add((name, (fixed | pinned & variable) & ~CONTROL, variable & ~pin_mask & ~CONTROL))
    return sorted(result, key=lambda form: (form[1] & 0xfff, form[1], form[2], form[0]))


def write(path, header):
    lines = [header, "", "/// Forms with the same opcode, modifiers and operand positions on sm80 and sm75.",
             "pub const SHARED_FORMS: &[(u128, u128)] = &["]
    for name, fixed, variable in shared():
        lines.extend([f"    // {name}", f"    (0x{fixed:032x}, 0x{variable:032x}),"])
    lines.append("];")
    lines.extend(["", "/// Forms of whole classes of sm80 (the same instruction on sm75, bit for bit, by the two",
                  "/// descriptions) whose translation ran right on a V100 (reference/ampere-evidence.txt): the",
                  "/// class, the values of its modifiers, and the bits of its other operands. Sorted by the",
                  "/// twelve low bits of the opcode.", "pub const WHOLE_CLASSES: &[(u128, u128)] = &["])
    previous = None
    for name, fixed, variable in evidenced_forms():
        if name != previous:
            lines.append(f"    // {name}")
            previous = name
        lines.append(f"    (0x{fixed:032x}, 0x{variable:032x}),")
    lines.append("];")
    for name, (fixed, variable) in canonical_forms().items():
        lines.extend(["", f"pub const {name}: (u128, u128) = (0x{fixed:032x}, 0x{variable:032x});"])
    Path(path).write_text("\n".join(lines) + "\n")
