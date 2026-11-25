"""Decode the narrow sm_80 forms with known Volta translations, or refuse the function."""
from functools import lru_cache
from ampere_forms import canonical_forms, forms, evidenced_forms

from machine import USCHED, WRITE_BARRIER, READ_BARRIER, WAIT

CONTROLS = 0x1ffff << USCHED

@lru_cache(maxsize=1)
def supported():
    return canonical_forms(), forms()


@lru_cache(maxsize=1)
def whole():
    """The forms of whole classes that ran right on a V100 (ampere-evidence.txt), by the twelve
    low bits of their opcode."""
    by_opcode = {}
    for _, fixed, variable in evidenced_forms():
        by_opcode.setdefault(fixed & 0xfff, []).append((fixed, variable))
    return by_opcode


def matches(bits, form):
    fixed, variable = form
    return bits & ~variable == fixed


def normalize(code, mc, relative, signed, refuse, helper=None):
    if not code or len(code) % 16:
        raise refuse('sm_80 code is empty or contains a partial instruction')
    canonical, shared = supported()
    classes = whole()
    result, steps, uses, operations = bytearray(), [], {}, {}
    joins, syncs = {}, []
    calls, branches, helper_writes = [], set(), set()
    copies, copied = {}, []
    returns = previous_move = None
    saw_return = False
    for at in range(0, len(code), 16):
        source = int.from_bytes(code[at:at+16], 'little')
        if source >> WRITE_BARRIER & 7 == 6 or source >> READ_BARRIER & 7 == 6:
            raise refuse(f'reserved sm_80 scoreboard at 0x{at:x}')
        bits = source & ~CONTROLS
        rd, addr, rb = [(source >> shift) & 255 for shift in (16, 24, 32)]
        descriptor = None
        load = matches(bits, canonical['LOAD'])
        constant = matches(bits, canonical['LOAD_CONST'])
        store = matches(bits, canonical['STORE'])
        offset = source >> 40 & 0xffffff
        # The size: bytes and halfwords, signed or not, then 32, 64 and 128 bits (0 to 6).
        size = source >> 73 & 7
        if load or constant or store:
            if size == 7:
                raise refuse(f'invalid size of an sm_80 global access at 0x{at:x}')
            if size >= 5:
                vector = rb if store else rd
                width = 4 if size == 6 else 2
                zero_store = store and size == 6 and vector == 255
                aligned_vector = vector % width == 0 and vector <= 256 - 2 * width
                if not (aligned_vector or zero_store) or addr % 2 or addr > 252:
                    raise refuse(f'invalid register group for sm_80 vector access at 0x{at:x}')
        if (load or constant or store) and (addr > 252 or addr % 2):
            # (Said here, before the instruction that stands for it is made: it has no such address.)
            raise refuse(f'unsupported sm_80 address register pair at 0x{at:x}')
        if load or constant:
            descriptor = rb & 63
            word = mc.assemble('ldg__sImmOffset', e='E', sz=size, sem='WEAK' if load else 'CONSTANT', sco='SYS', cop='EN',
                               Pu=7, Rd=rd, Ra=addr, Ra_offset=offset, Pg=source >> 12 & 7, Pg_not=source >> 15 & 1)
        elif store:
            descriptor = source >> 64 & 63
            word = mc.assemble('stg__sImmOffset', e='E', sz=size, sem='WEAK',
                               sco='SYS', cop='EN', Rb=rb, Ra=addr, Ra_offset=offset, Pg=source >> 12 & 7, Pg_not=source >> 15 & 1)
        elif matches(bits, canonical['BRA_MASK_ALL']):
            # Ampere makes the participation mask explicit; ~URZ includes every lane.
            word = mc.assemble('bra_', Pg=7, Pp=7, cond='DIV') | bits & canonical['BRA_MASK_ALL'][1]
        elif matches(bits, canonical['ATOM_ADD32']):
            descriptor = source >> 64 & 63
            word = mc.assemble('atomg__RaNonRZ', Rd=rd, Pu=7, Ra=addr, Ra_offset=offset,
                               Rb=rb, e='E', sz='32', op='ADD', cop='EN', sem='STRONG', sco='GPU')
        elif matches(bits, canonical['SHARED_INC']):
            if addr == 255:
                raise refuse(f'missing sm_80 shared increment index at 0x{at:x}')
            operations[at//16] = ('increment', offset)
            word = mc.assemble('sts__sImmOffset', Ra=addr, Rb=255, sz='_32',
                               Pg=source >> 12 & 7, Pg_not=source >> 15 & 1)
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
        elif matches(bits, canonical['BF16_ADD']):
            # What ptxas makes of fma.rn.bf16x2 by one: the sum of each half, rounded to bf16.
            # The f16 instruction of the same registers stands for it until it is rewritten.
            operations[at//16] = 'bf16add'
            word = mc.assemble('hfma2__RIR', Rd=rd, Ra=addr, Rc=source >> 64 & 255, Sb=0x3c00, Sb1=0x3c00)
        elif matches(bits, canonical['BF16_PACK']):
            # Two f32 rounded to bf16 in one register; the surrogate is that of the f16 form.
            operations[at//16] = 'bf16pack'
            word = mc.assemble('prmt__RuIR_RIR', Rd=rd, Ra=rb, Sb=0x5410, Rc=addr)
        elif matches(bits, canonical['PACK']):
            operations[at//16] = 'pack'
            word = mc.assemble('prmt__RuIR_RIR', Rd=rd, Ra=rb, Sb=0x5410, Rc=addr)
        elif any(matches(bits, form) for form in shared) or any(matches(bits, form) for form in classes.get(bits & 0xfff, ())):
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
                through = ops.get('Ra')
                if not inside or returns is None or target != 0 or conditional or (through != returns and through not in copies):
                    raise refuse(f'unproven sm_80 return at 0x{at:x}')
                if through != returns:
                    # Through a copy of the return register: nothing may enter between the two.
                    copied.append((copies[through], at//16))
                saw_return = True
                target,fall = None,False
            elif (target is not None and (target//16 >= helper) != inside) or (at//16 + 1 == helper and fall):
                raise refuse(f'sm_80 control flow crosses local function boundary at 0x{at:x}')
            if inside and ('R',returns) in d.writes:
                raise refuse(f'sm_80 local function overwrites its return register at 0x{at:x}')
            if inside:
                # A register that holds the return address: an unguarded move of the return register.
                for space, r in d.writes:
                    if space == 'R':
                        copies.pop(r, None)
                if (d.c.name == 'imad__RRR_RRR' and not conditional and ops['Ra'] == 255 and ops['Rb'] == 255
                        and ops['Rc'] == returns and ops['Rd'] < 255 and ops['fmt'] == 0
                        and not ops.get('Rc@negate') and d.writes == [('R', ops['Rd'])]):
                    copies[ops['Rd']] = at//16
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
            joins[at//16] = (ops['barReg'], target//16)
        elif d.mnemonic == 'BSYNC':
            syncs.append((at//16, ops['barReg']))
        steps.append((killed, initialized, successors))
    # The descriptors that nothing but their own load ever writes: once loaded, they hold it.
    constant = {q for q in set(uses.values()) if all(q in initialized for killed, initialized, _ in steps if q in killed)}
    if helper is not None:
        if not saw_return or not calls or any(first < target <= call for first, call in calls for target in branches):
            raise refuse('sm_80 local function lacks a proven call and return path')
        if any(copy < target <= back for copy, back in copied for target in branches):
            raise refuse('sm_80 return through a copy that a branch may skip')
        for _, call in calls:
            steps[call][0].update(helper_writes)
    # Track which convergence target is definite on each path leading to BSYNC.
    # A second use of the same barrier register starts a new region.
    if syncs:
        live, pending = {0: {}}, [0]
        callee_barriers = {b for i, (b, _) in joins.items() if helper is not None and i >= helper}
        callee_barriers.update(b for i,b in syncs if helper is not None and i >= helper)
        consumed = dict(syncs)
        call_sites = {call for _, call in calls}
        while pending:
            i = pending.pop()
            out = live[i].copy()
            if i in joins:
                b, target = joins[i]
                out[b] = target
            if i in consumed:
                out.pop(consumed[i], None)
            if i in call_sites:
                for b in callee_barriers:
                    out.pop(b, None)
            for j in steps[i][2]:
                after = out
                if i in joins and j == joins[i][1]:
                    after = out.copy()
                    after.pop(joins[i][0], None)
                known = {b:t for b,t in after.items() if live[j].get(b) == t} if j in live else after
                if live.get(j) != known:
                    live[j] = known
                    pending.append(j)
        # (By barrier register, then by place: the order in which the translator looks at them.)
        for i,b in sorted(syncs, key=lambda sync: (sync[1], sync[0])):
            if live.get(i, {}).get(b) != i+1:
                raise refuse(f'sm_80 BSYNC B{b} at 0x{16*i:x} does not fall through to its convergence target')
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
    # A thread reaches the place where the threads of a BSSY meet again by the BSYNC before it
    # (shown above to be the instruction before that place) or by a branch, not from the BSSY.
    # For a descriptor that is only ever written by its load, it is enough that each thread
    # passed a load of it: the paths of a thread, without the step from a BSSY to that place.
    passed, pending = {0: set()}, [0]
    while pending:
        i = pending.pop()
        killed, initialized, successors = steps[i]
        out = (passed[i] - killed) | initialized
        for j in successors:
            if i in joins and j == joins[i][1] and j != i + 1:
                continue
            joined = passed[j] & out if j in passed else out
            if passed.get(j) != joined:
                passed[j] = joined
                pending.append(j)
    for i, q in uses.items():
        if q not in state.get(i, set()) and not (q in constant and q in passed.get(i, set())):
            raise refuse(f'sm_80 global descriptor UR{q} is not established at 0x{16*i:x}')
    return result, operations
