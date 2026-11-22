//! Verify that a warp's non-coherent loads and 16 surviving accumulator
//! registers implement two affine Q4 matrix–vector products. All unknown
//! control flow, live values, nonlocal effects and different layouts reject
//! the transformation. The original body remains the dynamic fallback.
//!
//! For the measured layout, a warp owns a 32-column output tile. Each 64-K
//! group stores two 1,152-byte tables (packed four-bit weights, then BF16
//! scale/bias pairs). A one-row invocation can keep one output column in
//! each lane. This avoids distributing/recovering eight live outputs in
//! each of four lanes after every MMA. The verifier interprets the original
//! scalar and warp operations for every lane and K group. Memory addresses
//! remain symbolic in the expert, item, member and row. The generated path
//! retains the original grid, argument ABI, epilogue and store owners.
//!
//! Concrete launch and argument guards cover the batch-one, 2,048 by 512
//! case. Externally loaded guards must dominate the cut and be immutable on
//! *all* incoming paths. Input range checks guard the opt-in BF16 tree's
//! arithmetic contract. Until those checks pass all results are private.
mod emit;
mod eval;
mod program;
mod symbolic;
mod values;

use eval::{Bindings, Frame};
use program::{Arg, Program, Register};
use std::collections::{BTreeMap, HashSet};
use symbolic::{Symbolic, Term};
use values::{Value, Word};
const MMA: &str = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32";
#[derive(Debug)]
struct Transfer<'a> {
    at: usize,
    label: &'a str,
    params: Vec<&'a str>,
    inputs: Bindings,
    outputs: BTreeMap<Register, (u8, u8)>,
    names: Vec<&'a str>,
}

fn shuffle(lanes: &mut [Frame], args: &[Arg]) -> Option<()> {
    let [Arg::Tuple(dest), source, idx, clamp, mask] = args else {
        return None;
    };
    let [dst, pred] = dest.as_slice() else { return None };
    let (d, p) = (dst.reg()?, pred.reg()?);
    let mut values = Vec::with_capacity(32);
    let hop = usize::try_from(lanes.first()?.val(idx).scalar()?).ok()?;
    if !matches!(hop, 1 | 2) {
        return None;
    }
    for l in lanes.iter() {
        if l.val(clamp).scalar() != Some(31)
            || !matches!(l.val(mask).scalar(), Some(-1 | 4_294_967_295))
            || l.val(idx).scalar() != Some(hop as i64)
        {
            return None;
        }
        values.push(l.val(source));
    }
    for l in lanes {
        l.put(d, values[l.lane ^ hop]);
        l.put(p, Value::int(1));
    }
    Some(())
}
fn mma(lanes: &mut [Frame], args: &[Arg], math: &mut Symbolic, calls: &mut [u8; 256]) -> Option<()> {
    let [Arg::Tuple(ds), Arg::Tuple(a), Arg::Tuple(b), Arg::Tuple(c)] = args else {
        return None;
    };
    if ds.len() != 4 || a.len() != 4 || b.len() != 2 || c.len() != 4 {
        return None;
    }
    let mut a_terms = [[[0u16; 2]; 2]; 4];
    for l in lanes.iter() {
        for (i, r) in a.iter().enumerate() {
            let w = l.val(r).bits()?;
            if l.lane >= 4 || i % 2 == 1 {
                if w != Word::literal(0) {
                    return None;
                }
                continue;
            }
            for (h, value) in a_terms[l.lane][i / 2].iter_mut().enumerate() {
                let Term::Input(k) = math.term(w, h * 16, 16)? else {
                    return None;
                };
                *value = k;
            }
        }
    }
    let base = a_terms[0][0][0];
    let g = (base / 64) as u8;
    let part = (base % 64) as u8;
    if !matches!(part, 0 | 4 | 8 | 12) {
        return None;
    }
    for (q, a) in a_terms.iter().enumerate() {
        for (h, pair) in a.iter().enumerate() {
            for (j, &k) in pair.iter().enumerate() {
                if k != base + (16 * q + 2 * h + j) as u16 {
                    return None;
                }
            }
        }
    }
    let mut columns = [(0u8, 0u8); 8];
    for (col, slot) in columns.iter_mut().enumerate() {
        let mut id = None;
        for q in 0..4 {
            for (h, r) in b.iter().enumerate() {
                for (j, &expected_k) in a_terms[q][h].iter().enumerate() {
                    let w = lanes[col * 4 + q].val(r).bits()?;
                    let Term::Nibble {
                        group,
                        table,
                        column,
                        k,
                    } = math.term(w, j * 16, 16)?
                    else {
                        return None;
                    };
                    if group != g || u16::from(group) * 64 + u16::from(k) != expected_k {
                        return None;
                    }
                    if id.replace((table, column)).is_some_and(|old| old != (table, column)) {
                        return None;
                    }
                }
            }
        }
        *slot = id?;
    }
    let table = columns[0].0;
    if columns.iter().any(|x| x.0 != table) {
        return None;
    }
    let counter = calls.get_mut(usize::from(g) * 8 + usize::from(table) * 4 + usize::from(part / 4))?;
    *counter = counter.checked_add(1)?;
    for l in lanes {
        let old: Vec<_> = c.iter().map(|r| l.val(r).bits()).collect::<Option<_>>()?;
        let mut written = HashSet::new();
        for (i, d) in ds.iter().enumerate() {
            let r = d.reg()?;
            if !written.insert(r) {
                return None;
            }
            let zero = Word::literal(0);
            let new = if l.lane / 4 + (i / 2) * 8 == 0 {
                let (table, column) = columns[(l.lane % 4) * 2 + i % 2];
                if part == 0 {
                    if old[i] != zero {
                        return None;
                    }
                } else if math.term(old[i], 0, 32)
                    != Some(Term::Dot {
                        group: g,
                        table,
                        column,
                        part: part - 4,
                    })
                {
                    return None;
                }
                math.word(Term::Dot {
                    group: g,
                    table,
                    column,
                    part,
                })?
            } else {
                if old[i] != zero {
                    return None;
                }
                zero
            };
            l.put(r, Value::Bits(new));
        }
    }
    Some(())
}
fn prove<'a>(program: Program<'a>) -> Option<Transfer<'a>> {
    let p = &program;
    let first = p
        .statements
        .iter()
        .position(|s| s.op.starts_with("ld.global") && s.op.contains(".v4."))?;
    if p.statements[first].guard.is_some() {
        return None;
    }
    let backs: Vec<_> = p
        .statements
        .iter()
        .enumerate()
        .filter_map(|(i, s)| match s.args.as_slice() {
            [Arg::Label(t)] if *t < i => Some(i),
            _ => None,
        })
        .collect();
    let [inner, _outer] = backs.as_slice() else { return None };
    let end = inner + 1;
    let label = *p.labels.iter().find(|(_, v)| **v == end)?.0;
    let mut math = Symbolic::default();
    let mut lanes: Vec<_> = (0..32).map(|t| Frame::new(p, t)).collect();
    let mut calls = [0u8; 256];
    loop {
        let mut reached = None;
        for l in &mut lanes {
            let stop = l.advance(p, first, end, &mut math)?;
            if reached.replace(stop).is_some_and(|prior| prior != stop) {
                return None;
            }
        }
        let Some(pc) = reached? else { break };
        let s = &p.statements[pc];
        if s.op == MMA {
            mma(&mut lanes, &s.args, &mut math, &mut calls)?
        } else {
            shuffle(&mut lanes, &s.args)?
        }
    }
    if calls.iter().any(|&n| n != 4) || lanes.iter().any(|l| !l.verified_memory()) {
        return None;
    }
    let mut outputs = BTreeMap::new();
    for l in &lanes {
        if l.lane >= 4 {
            if !l.epilogue_values.is_empty() {
                return None;
            }
            continue;
        }
        let wanted = math.expected_outputs(l.lane as u8)?;
        if wanted.len() != 16 || l.epilogue_values.len() != 16 {
            return None;
        }
        let mut found = HashSet::new();
        for (r, w) in &l.epilogue_values {
            let &(t, c) = wanted.get(w)?;
            if !found.insert((t, c)) {
                return None;
            }
            if l.lane == 0 {
                outputs.insert(*r, (t, c));
            } else if outputs.get(r) != Some(&(t, c.checked_sub(2 * l.lane as u8)?)) {
                return None;
            }
        }
    }
    let b = lanes[0].binding.clone()?;
    let outer_head = match p.statements[*backs.last()?].args.as_slice() {
        [Arg::Label(to)] => *to,
        _ => return None,
    };
    for (index, reg) in [b.counts, b.expert, b.first, b.rows].into_iter().enumerate() {
        let (source, origin) = *lanes[0].origins.get(index)?;
        if source != reg
            || !p.immutable_guard(reg, origin, first, 0)
            || (index > 0 && !p.immutable_guard(reg, origin, first, outer_head))
        {
            return None;
        }
    }
    // At count==1, lanes 0..3 read the same member. Its register is
    // initialized to zero on the other branch, so (unlike the four
    // dominating metadata loads) it may have a definition *before* that load.
    let (member, load) = *lanes[0].origins.get(4)?;
    let inner_target = match p.statements[*inner].args.as_slice() {
        [Arg::Label(to)] => *to,
        _ => return None,
    };
    if member != b.member || load >= first || inner_target <= first || !p.unchanged(member, load + 1, first) {
        return None;
    }
    for l in &lanes {
        let (counts, rows, e, f, m) = l.guarded_indices?;
        if (counts, rows, e, f) != (b.counts, b.rows, b.expert, b.first) || m.is_some_and(|id| id != b.member) {
            return None;
        }
    }
    if ![b.counts, b.rows, b.expert, b.first, b.member, b.row, b.weights]
        .into_iter()
        .chain(outputs.keys().copied())
        .all(|r| p.roots[r])
    {
        return None;
    }
    Some(Transfer {
        at: p.statements[first].position,
        label,
        params: p.params.clone(),
        inputs: b,
        outputs,
        names: p.names.clone(),
    })
}

pub(super) fn rewrite(signature: &str, text: &str) -> Option<String> {
    // Keep the verifier off the normal path for unrelated modules.
    if !text.contains(MMA) || !text.contains("fma.rn.bf16x2") || !text.contains("ld.global.nc.L1::no_allocate.v4.u32") {
        return None;
    }
    let clean = crate::scan::blank_comments(text);
    let p = Program::parse(signature, &clean)?;
    let transfer = prove(p)?;
    let code = emit::write(&transfer);
    let mut out = String::with_capacity(text.len() + code.len());
    out.push_str(&text[..transfer.at]);
    out.push_str(&code);
    out.push_str(&text[transfer.at..]);
    Some(out)
}

#[cfg(test)]
mod tests;
