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
    "LOAD16": "ldg_memdesc__Ra64 Rd Ra Ra_URb | Pg=PT e=E cop=EN sz=U16 Pu=PT Pnz=PT input_reg_sz_64_dist=64 sem=CONSTANT",
    "LOAD128": "ldg_memdesc__Ra64 Rd Ra Ra_URb Ra_offset | Pg=PT e=E cop=EN sz=128 Pu=PT Pnz=PT input_reg_sz_64_dist=64 sem=WEAK",
    "STORE128": "stg_memdesc__Ra64 Ra Rb Ra_URc Ra_offset | Pg=PT e=E cop=EN sz=128 input_reg_sz_64_dist=64 sem=WEAK",
    "STORE16": "stg_memdesc__Ra64 Ra Rb Ra_URc | Pg=PT e=E cop=EN sz=U16 input_reg_sz_64_dist=64 sem=WEAK",
    "STORE32": "stg_memdesc__Ra64 Pg Pg.not Ra Rb Ra_URc Ra_offset | e=E cop=EN sz=32 input_reg_sz_64_dist=64 sem=WEAK",
    "LOAD32": "ldg_memdesc__Ra64 Rd Ra Ra_URb Ra_offset | Pg=PT e=E cop=EN sz=32 Pu=PT Pnz=PT input_reg_sz_64_dist=64 sem=WEAK",
    "LOAD32_CONST": "ldg_memdesc__Ra64 Rd Ra Ra_URb Ra_offset | Pg=PT e=E cop=EN sz=32 Pu=PT Pnz=PT input_reg_sz_64_dist=64 sem=CONSTANT",
    "BRA_MASK_ALL": "bra_uniform_ sImm | Pg=PT Pp=PT cond=DIV URb=URZ URb.invert=1",
    "ATOM_ADD32": "atomg_uniform__memdesc Rd Ra Ra_URc Rb Ra_offset | Pg=PT e=E op=ADD cop=EN sz=32 Pu=PT input_reg_sz_64_dist=64 sem=STRONG sco=GPU",
    "SHARED_INC": "atoms_arrive__popcinc Pg Pg.not Ra Ra_offset | Rd=RZ stride=X4 Ra_URc=URZ op=POPC.INC sz=32",
    "COPY128": "ldgsts__desc_RRU Pg Pg.not Rb Rb_offset Ra_URc Ra Ra_offset | loc=ACCESS cop=EN sz=128 input_reg_sz_64_dist=64 Pnz=PT sem=WEAK",
    "COMMIT": "ldgdepbar_ | Pg=PT",
    "HALF_CONSTANT": "hfma2_mma__RRI_norelu Rd Sc Sc1 | Pg=PT Ra=RZ Ra.negate=1 Rb=RZ",
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
        if kind == "operand":
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


def write(path, header):
    lines = [header, "", "/// Forms with the same opcode, modifiers and operand positions on sm80 and sm75.",
             "pub const SHARED_FORMS: &[(u128, u128)] = &["]
    for name, fixed, variable in shared():
        lines.extend([f"    // {name}", f"    (0x{fixed:032x}, 0x{variable:032x}),"])
    lines.append("];")
    for name, (fixed, variable) in canonical_forms().items():
        lines.extend(["", f"pub const {name}: (u128, u128) = (0x{fixed:032x}, 0x{variable:032x});"])
    Path(path).write_text("\n".join(lines) + "\n")
