"""Decode the narrow sm_80 forms with known Volta translations, or refuse the function."""
from functools import lru_cache
from ampere_forms import canonical_forms, forms

from machine import USCHED, WRITE_BARRIER, READ_BARRIER

CONTROLS = 0x1ffff << USCHED

@lru_cache(maxsize=1)
def supported():
    return canonical_forms(), forms()


def matches(bits, form):
    fixed, variable = form
    return bits & ~variable == fixed


def normalize(code, mc, relative, signed, refuse):
    if not code or len(code) % 16:
        raise refuse('sm_80 code is empty or contains a partial instruction')
    canonical, shared = supported()
    result, steps, uses = bytearray(), [], {}
    joins, syncs = {}, []
    for at in range(0, len(code), 16):
        source = int.from_bytes(code[at:at+16], 'little')
        if source >> WRITE_BARRIER & 7 == 6 or source >> READ_BARRIER & 7 == 6:
            raise refuse(f'reserved sm_80 scoreboard at 0x{at:x}')
        bits = source & ~CONTROLS
        rd, addr, rb = [(source >> shift) & 255 for shift in (16, 24, 32)]
        descriptor = None
        if matches(bits, canonical['LOAD16']):
            descriptor = rb & 63
            word = mc.assemble('ldg__sImmOffset', e='E', sz='U16', sem='CONSTANT', sco='SYS', cop='EN', Pu=7, Rd=rd, Ra=addr)
        elif matches(bits, canonical['STORE16']) or matches(bits, canonical['STORE32']):
            descriptor = source >> 64 & 63
            word = mc.assemble('stg__sImmOffset', e='E', sz='U16' if matches(bits, canonical['STORE16']) else '_32', sem='WEAK',
                               sco='SYS', cop='EN', Rb=rb, Ra=addr)
        elif matches(bits, canonical['PACK']):
            word = mc.assemble('f2f_f32_downconvert__RRR_RRR', Rd=rd, Rb=rb, **{'dstfmt.srcfmt': 'F16.F32'})
        elif any(matches(bits, form) for form in shared):
            word = bits
        else:
            raise refuse(f'unsupported sm_80 instruction {source:032x} at 0x{at:x}')
        if descriptor is not None:
            if descriptor > 60 or descriptor % 2 or addr > 252 or addr % 2:
                raise refuse(f'unsupported sm_80 address register pair at 0x{at:x}')
            uses[at//16] = descriptor
        word = (word & ~CONTROLS) | (source & CONTROLS)
        d = mc.decode(75, word)
        if d is None:
            raise refuse(f'no decoded instruction for the sm_80 form at 0x{at:x}')
        result.extend(word.to_bytes(16, 'little'))
        killed = {r & ~1 for space, r in d.writes if space == 'U'}
        ops = d.ops
        initialized = set()
        q = ops.get('URd', 63)
        if (d.c.name == 'uldc_const__RCR' and q <= 60 and q % 2 == 0 and ops['sz'] == 5
                and ops['UPg'] == 7 and not ops['UPg@not'] and ops['Sa_bank'] == 0 and ops['Sa_addr'] == 0x118):
            initialized.add(q)
        successors = []
        conditional = (ops.get('Pg', 7) != 7 or ops.get('Pg@not', 0) or ops.get('Pp', 7) != 7
                       or ops.get('Pp@not', 0) or ops.get('cond', 0))
        if conditional or d.mnemonic not in ('BRA', 'EXIT'):
            if at+16 < len(code):
                successors.append(at//16 + 1)
        rel = relative(mc, 75, d)
        if rel:
            field, scale = rel
            target = at+16 + signed(field, word)*scale
            if not 0 <= target < len(code) or target % 16:
                raise refuse(f'sm_80 branch at 0x{at:x} leaves its function')
            successors.append(target//16)
        if d.mnemonic == 'BSSY':
            barrier = ops['barReg']
            if barrier in joins and joins[barrier] != target//16:
                raise refuse(f'multiple sm_80 convergence targets for B{barrier}')
            joins[barrier] = target//16
        elif d.mnemonic == 'BSYNC':
            syncs.append((at//16, ops['barReg']))
        steps.append((killed, initialized, successors))
    for i, barrier in syncs:
        if joins.get(barrier) != i+1:
            raise refuse(f'sm_80 BSYNC B{barrier} at 0x{16*i:x} does not fall through to its convergence target')
    state, pending = {0: set()}, [0]
    while pending:
        i = pending.pop()
        killed, initialized, successors = steps[i]
        out = (state[i] - killed) | initialized
        for j in successors:
            joined = state[j] & out if j in state else out
            if state.get(j) != joined:
                state[j] = joined
                pending.append(j)
    for i, q in uses.items():
        if q not in state.get(i, set()):
            raise refuse(f'sm_80 global descriptor UR{q} is not established at 0x{16*i:x}')
    return result
