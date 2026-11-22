"""Decode the narrow sm_80 forms with known Volta translations, or refuse the function."""
from functools import lru_cache
from ampere_forms import canonical_forms, forms

from machine import USCHED, WRITE_BARRIER, READ_BARRIER, WAIT

CONTROLS = 0x1ffff << USCHED

@lru_cache(maxsize=1)
def supported():
    return canonical_forms(), forms()


def matches(bits, form):
    fixed, variable = form
    return bits & ~variable == fixed


def normalize(code, mc, relative, signed, refuse, helper=None):
    if not code or len(code) % 16:
        raise refuse('sm_80 code is empty or contains a partial instruction')
    canonical, shared = supported()
    result, steps, uses, operations = bytearray(), [], {}, {}
    joins, syncs = {}, []
    calls, branches, helper_writes = [], set(), set()
    returns = previous_move = None
    saw_return = False
    for at in range(0, len(code), 16):
        source = int.from_bytes(code[at:at+16], 'little')
        if source >> WRITE_BARRIER & 7 == 6 or source >> READ_BARRIER & 7 == 6:
            raise refuse(f'reserved sm_80 scoreboard at 0x{at:x}')
        bits = source & ~CONTROLS
        rd, addr, rb = [(source >> shift) & 255 for shift in (16, 24, 32)]
        descriptor = None
        load128 = matches(bits, canonical['LOAD128'])
        store128 = matches(bits, canonical['STORE128'])
        offset = source >> 40 & 0xffffff
        if load128 or store128:
            vector = rd if load128 else rb
            if vector % 4 or vector > 248 or addr % 2 or addr > 252:
                raise refuse(f'invalid register group for sm_80 vector access at 0x{at:x}')
        if matches(bits, canonical['LOAD16']) or load128:
            descriptor = rb & 63
            word = mc.assemble('ldg__sImmOffset', e='E', sz='_128' if load128 else 'U16',
                               sem='WEAK' if load128 else 'CONSTANT', sco='SYS', cop='EN',
                               Pu=7, Rd=rd, Ra=addr, Ra_offset=offset)
        elif matches(bits, canonical['STORE16']) or matches(bits, canonical['STORE32']) or store128:
            descriptor = source >> 64 & 63
            size = '_128' if store128 else 'U16' if matches(bits, canonical['STORE16']) else '_32'
            word = mc.assemble('stg__sImmOffset', e='E', sz=size, sem='WEAK',
                               sco='SYS', cop='EN', Rb=rb, Ra=addr, Ra_offset=offset)
        elif matches(bits, canonical['HALF_CONSTANT']):
            # RN(-0 * +0 + c) is c for finite FP16 operands, including signed zero.
            pair = source >> 32 & 0xffffffff
            if any((h & 0x7c00) == 0x7c00 for h in (pair, pair >> 16)):
                raise refuse(f'nonfinite sm_80 half constant at 0x{at:x}')
            word = mc.assemble('imad__RRsI_RRI', Rd=rd, Ra=255, Rb=255, Sc=pair)
        elif matches(bits, canonical['COPY128']):
            if addr > 252 or addr % 2:
                raise refuse(f'invalid address for sm_80 asynchronous copy at 0x{at:x}')
            descriptor = source >> 64 & 63
            source_offset = source >> 32 & 0xfff
            if source_offset & 0x800:
                source_offset -= 0x1000
            shared_offset = source >> 44 & 0xfffff
            if shared_offset & 0x80000:
                shared_offset -= 0x100000
            if rd == 255 and shared_offset < 0:
                raise refuse(f'negative absolute shared address at 0x{at:x}')
            operations[at // 16] = ('copy128', source_offset, shared_offset)
            # The scalar store represents the two input addresses and guard for liveness.
            word = mc.assemble('stg__sImmOffset', e='E', sz='_32', sem='WEAK', sco='SYS',
                               cop='EN', Ra=addr, Rb=rd, Pg=source >> 12 & 7, **{'Pg_not':source >> 15 & 1})
        elif matches(bits, canonical['COMMIT']):
            word = mc.assemble('membar_', sem='SC', sco='CTA')
        elif matches(bits, canonical['PACK']):
            operations[at//16] = 'pack'
            word = mc.assemble('prmt__RuIR_RIR', Rd=rd, Ra=rb, Sb=0x5410, Rc=addr)
        elif any(matches(bits, form) for form in shared):
            word = bits
        else:
            raise refuse(f'unsupported sm_80 instruction {source:032x} at 0x{at:x}')
        if descriptor is not None:
            if descriptor > 60 or descriptor % 2 or addr > 252 or addr % 2:
                raise refuse(f'unsupported sm_80 address register pair at 0x{at:x}')
            uses[at//16] = descriptor
        if matches(bits,canonical['COMMIT']) or matches(bits,canonical['COPY128']):
            word |= 63 << WAIT
        else:
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
        fall = conditional or d.mnemonic not in ('BRA', 'EXIT', 'RET')
        rel = relative(mc, 75, d)
        target = None
        if rel:
            field, scale = rel
            target = at+16 + signed(field, word)*scale
            if not 0 <= target < len(code) or target % 16:
                raise refuse(f'sm_80 branch at 0x{at:x} leaves its function')
        if d.mnemonic in ('CALL', 'RET') or helper is not None:
            if helper is None:
                raise refuse('sm_80 CALL or RET without a local function symbol')
            inside = at//16 >= helper
            if d.mnemonic == 'CALL':
                if previous_move is None:
                    raise refuse(f'unproven sm_80 return address at 0x{at:x}')
                from_at, reg, back = previous_move
                if back != at + 16 or inside or target != helper*16 or at//16+1 >= helper or conditional or (returns is not None and returns != reg):
                    raise refuse(f'unsupported sm_80 local call at 0x{at:x}')
                calls.append((from_at, at//16))
                returns = reg
            elif d.mnemonic == 'RET':
                if not inside or returns is None or ops.get('Ra') != returns or target != 0 or conditional:
                    raise refuse(f'unproven sm_80 return at 0x{at:x}')
                saw_return = True
                target,fall = None,False
            elif (target is not None and (target//16 >= helper) != inside) or (at//16 + 1 == helper and fall):
                raise refuse(f'sm_80 control flow crosses local function boundary at 0x{at:x}')
            if inside and ('R',returns) in d.writes:
                raise refuse(f'sm_80 local function overwrites its return register at 0x{at:x}')
            if inside:
                helper_writes |= killed
            if d.c.name == 'mov__RI' and not conditional:
                previous_move = (at//16, ops['Rd'], ops['Sb']) if ops['Rd'] < 255 else None
            elif d.kind == 'BRU' or (previous_move is not None and ('R',previous_move[1]) in d.writes):
                previous_move = None
        if fall and at + 16 < len(code):
            successors.append(at//16 + 1)
        if target is not None:
            successors.append(target//16)
            branches.add(target//16)
        if d.mnemonic == 'BSSY':
            barrier = ops['barReg']
            if barrier in joins and joins[barrier] != target//16:
                raise refuse(f'multiple sm_80 convergence targets for B{barrier}')
            joins[barrier] = target//16
        elif d.mnemonic == 'BSYNC':
            syncs.append((at//16, ops['barReg']))
        steps.append((killed, initialized, successors))
    if helper is not None:
        if not saw_return or not calls or any(first < target <= call for first, call in calls for target in branches):
            raise refuse('sm_80 local function lacks a proven call and return path')
        for _, call in calls:
            steps[call][0].update(helper_writes)
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
    return result, operations
