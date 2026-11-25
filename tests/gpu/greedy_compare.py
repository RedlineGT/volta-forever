"""What vllm.sh and sglang.sh do with a greedy text and the CPU's reference for it.

A greedy text is the CPU's, token for token, or it leaves it somewhere. Where it leaves it, the
question is whether the CPU itself had a choice there: `vllm-ref.py` records, for each token of
its reference, by how much its logit led the next one's (float32). Two right computations of a
model in float16 differ by 1e-2 to 7e-2 in a log-probability, so a token whose
lead on the CPU is under NEAR (0.1) can be the other one in float16 without anything being wrong:
such a first difference is a NEAR TIE, said as such with the lead, and what follows it cannot be
compared (another prefix). Any other first difference is a DIFFERENCE.

    compare(record, ids=None, text=None, near=NEAR) -> (kind, detail): kind is 'match', 'near' or
        'differ'; `near`: the lead under which a first difference is a near tie
    conclude(kinds) -> the verdict: ALL_MATCH, MATCH_BUT_NEAR_TIES n, or SOME_DIFFER, which the
        programs print on a line of its own after `verdict: ` (vllm.sh and sglang.sh read that
        line, the last of them, and no other)
"""

NEAR = 0.1
# For a model in bfloat16 against the CPU's reference in bfloat16 (`near` of `compare`): 8 bits of
# significand, so a logit between 8 and 16 is a multiple of 0.0625, and between 16 and 32 of
# 0.125; the reference's leads are such multiples, and two right computations (operations
# rounded one by one, as PyTorch has them, or once, as a fused kernel does) put a logit a unit or
# two apart. Measured on a V100, 2026-11-25, with the model of vllm.sh: vLLM with its compilation
# on, whose text is the CPU's bfloat16 text in every token for two prompts of three, leaves it
# in the third where the CPU led by 0.125; with its compilation off, at leads of 0, 0.0625 and
# 0.125. A lead of 0.25 or more is not a near tie.
NEAR_BF16 = 0.25


def compare(record, ids=None, text=None, near=NEAR):
    reference, margins = record['token_ids'], record.get('margins')
    if ids is not None:
        ids = list(ids)
        at = next((k for k, (x, y) in enumerate(zip(ids, reference)) if x != y), None)
        if at is None and len(ids) == len(reference):
            return 'match', '%d tokens' % len(ids)
        if at is None:
            return 'differ', '%d tokens where the reference has %d' % (len(ids), len(reference))
        where = 'token %d' % at
    else:
        if text == record['text']:
            return 'match', '%d tokens (by the text)' % len(reference)
        char = next((k for k, (x, y) in enumerate(zip(text, record['text'])) if x != y), min(len(text), len(record['text'])))
        # The token the first different character is in, by the reference's tokens as text.
        at, seen = None, 0
        for k, piece in enumerate(record.get('token_texts') or []):
            seen += len(piece)
            if char < seen:
                at = k
                break
        if at is None:
            return 'differ', 'at character %d: %r, the reference %r' % (char, text[max(0, char - 30):char + 40], record['text'][max(0, char - 30):char + 40])
        where = 'token %d (character %d: %r, the reference %r)' % (at, char, text[max(0, char - 25):char + 25], record['text'][max(0, char - 25):char + 25])
    if margins and at < len(margins) and margins[at] < near:
        return 'near', '%s, where the reference led by %.4f in logit on the CPU: a near tie; %d tokens the same before it, the rest not comparable' % (where, margins[at], at)
    lead = ' (the reference led by %.4f there)' % margins[at] if margins and at < len(margins) else ''
    return 'differ', where + lead


def conclude(kinds):
    if any(k == 'differ' for k in kinds):
        return 'SOME_DIFFER'
    near = sum(k == 'near' for k in kinds)
    return 'MATCH_BUT_NEAR_TIES %d' % near if near else 'ALL_MATCH'
