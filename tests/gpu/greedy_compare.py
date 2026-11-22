"""What vllm.sh and sglang.sh do with a greedy text and the CPU's reference for it.

A greedy text is the CPU's, token for token, or it leaves it somewhere. Where it leaves it, the
question is whether the CPU itself had a choice there: `vllm-ref.py` records, for each token of
its reference, by how much its logit led the next one's (float32). Two right computations of a
model in float16 differ by 1e-2 to 7e-2 in a log-probability, so a token whose
lead on the CPU is under NEAR (0.1) can be the other one in float16 without anything being wrong:
such a first difference is a NEAR TIE, said as such with the lead, and what follows it cannot be
compared (another prefix). Any other first difference is a DIFFERENCE.

    compare(record, ids=None, text=None) -> (kind, detail): kind is 'match', 'near' or 'differ'
    conclude(kinds) -> the verdict: ALL_MATCH, MATCH_BUT_NEAR_TIES n, or SOME_DIFFER, which the
        programs print on a line of its own after `verdict: ` (vllm.sh and sglang.sh read that
        line, the last of them, and no other)
"""

NEAR = 0.1


def compare(record, ids=None, text=None):
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
    if margins and at < len(margins) and margins[at] < NEAR:
        return 'near', '%s, where the reference led by %.4f in logit on the CPU: a near tie; %d tokens the same before it, the rest not comparable' % (where, margins[at], at)
    lead = ' (the reference led by %.4f there)' % margins[at] if margins and at < len(margins) else ''
    return 'differ', where + lead


def conclude(kinds):
    if any(k == 'differ' for k in kinds):
        return 'SOME_DIFFER'
    near = sum(k == 'near' for k in kinds)
    return 'MATCH_BUT_NEAR_TIES %d' % near if near else 'ALL_MATCH'
