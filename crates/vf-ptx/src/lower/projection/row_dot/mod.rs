//! Symbolically execute one-row BF16 dot products. A result is observable only
//! after proving each original FMA operand, in order, for every producing lane.
//! All lanes participate in the check of shared writes, reads and barriers.
//! The guarded layout has one active input row and sixteen output columns per
//! CTA. Verify both null and live row metadata (one valid row), and separately
//! prove a full output tile or a final tile with one column. The physical tile
//! index and the number of additional columns remain independent affine terms.
//! The original program handles every other case. The block keeps its original
//! static shared storage, output address, launch dimensions and parameter ABI.
mod emit;
mod value;
use super::program::{Arg, Program};
use Value::{Dot, Float, Half, Int, Ptr, Unknown};
use value::{Linear, Source, Space, Value};
const TYPES: [&str; 8] = [".u64", ".u32", ".u64", ".u64", ".u32", ".u32", ".u32", ".u64"];
const DEPTHS: [i64; 2] = [1024, 2048];
const TILE: usize = 16;

fn kind(p: &Program<'_>, a: &Arg, t: u8) -> bool {
    match a {
        Arg::Reg(r) => p.widths[*r] == t,
        Arg::Imm(x) => match t {
            64 => true,
            32 => i32::try_from(*x).is_ok(),
            _ => *x >= 0 && *x < (1i64 << t),
        },
        Arg::Shared(_) => t == 32,
        _ => false,
    }
}
fn typed(p: &Program<'_>) -> bool {
    p.statements.iter().all(|s| {
        let a = &s.args;
        match (s.op, &a[..]) {
            ("bra" | "bra.uni", [Arg::Label(_)]) => s.guard.is_none() || s.op == "bra",
            ("bar.sync", [Arg::Imm(0)]) | ("ret", []) => s.guard.is_none(),
            ("ld.param.u32" | "ld.param.u64", [Arg::Reg(d), Arg::Param(i)]) => {
                p.widths[*d] == if TYPES[*i] == ".u32" { 32 } else { 64 }
                    && (s.op == "ld.param.u32") == (TYPES[*i] == ".u32")
            }
            ("ld.global.nc.u16" | "ld.global.nc.u32" | "ld.shared.u16", [Arg::Reg(d), Arg::Mem(base, _)]) => {
                p.widths[*d] == if s.op.ends_with("u16") { 16 } else { 32 }
                    && p.widths[*base] == if s.op == "ld.shared.u16" { 32 } else { 64 }
            }
            ("st.global.f32" | "st.shared.u16", [Arg::Mem(base, _), Arg::Reg(src)]) => {
                p.widths[*base] == if s.op == "st.shared.u16" { 32 } else { 64 }
                    && p.widths[*src] == if s.op == "st.shared.u16" { 16 } else { 32 }
            }
            ("mov.b32", [Arg::Reg(v), Arg::Tuple(ts)]) => {
                p.widths[*v] == 32 && matches!(&ts[..], [Arg::Imm(0), Arg::Reg(_)]) && kind(p, &ts[1], 16)
            }
            (op, [Arg::Reg(d), src @ ..]) => {
                let (dw, sw): (u8, &[u8]) =
                    match op {
                        "mov.u32" | "mov.f32" | "not.b32" => (32, &[32]),
                        "cvta.to.global.u64" => (64, &[64]),
                        "cvt.s64.s32" => (64, &[32]),
                        "cvt.u32.u64" => (32, &[64]),
                        "cvt.rn.bf16.f32" => (16, &[32]),
                        "add.s32" | "sub.s32" | "and.b32" | "shl.b32" | "shr.s32" | "shr.u32" | "min.s32"
                        | "max.s32" => (32, &[32, 32]),
                        "add.s64" | "mul.lo.s64" => (64, &[64, 64]),
                        "mul.wide.s32" => (64, &[32, 32]),
                        "shl.b64" => (64, &[64, 32]),
                        "mad.lo.s32" | "fma.rn.f32" => (32, &[32, 32, 32]),
                        "and.pred" | "or.pred" => (1, &[1, 1]),
                        "setp.eq.s64" => (1, &[64, 64]),
                        "setp.eq.s32" | "setp.ne.s32" | "setp.lt.s32" | "setp.ge.s32" | "setp.gt.s32"
                        | "setp.lt.u32" => (1, &[32, 32]),
                        _ => return false,
                    };
                p.widths[*d] == dw && src.len() == sw.len() && src.iter().zip(sw).all(|(v, w)| kind(p, v, *w))
            }
            _ => false,
        }
    })
}
struct Tile {
    k: i64,
    valid: usize,
    metadata: bool,
    shared: [Vec<Option<Value>>; 2],
    writes: [Vec<Option<Value>>; 2],
    reads: [Vec<bool>; 2],
    outputs: [bool; TILE],
    remaining: usize,
}
impl Tile {
    fn new(p: &Program<'_>, k: i64, valid: usize, metadata: bool) -> Self {
        Self {
            k,
            valid,
            metadata,
            shared: std::array::from_fn(|i| vec![None; p.shared[i].1 / 2]),
            writes: std::array::from_fn(|i| vec![None; p.shared[i].1 / 2]),
            reads: std::array::from_fn(|i| vec![false; p.shared[i].1 / 2]),
            outputs: [false; TILE],
            remaining: 6_000_000,
        }
    }
    fn barrier(&mut self) -> Option<()> {
        for i in 0..2 {
            for (slot, (read, w)) in self.shared[i]
                .iter_mut()
                .zip(self.reads[i].iter_mut().zip(&mut self.writes[i]))
            {
                if let Some(value) = w.take() {
                    if *read {
                        return None;
                    }
                    *slot = Some(value)
                }
                *read = false;
            }
        }
        Some(())
    }
    fn param(&self, i: usize) -> Value {
        let zero = Linear::imm(0);
        match i {
            0 => Ptr(Space::Generic(0), zero),
            1 => Unknown,
            2 => Ptr(Space::Generic(2), zero),
            3 => Ptr(Space::Generic(3), zero),
            4 => Value::number(1),
            5 => Value::number(self.k),
            6 => Int(Linear {
                c: self.valid as i64,
                tile: TILE as i64,
                tail: i64::from(self.valid == TILE),
            }),
            7 if self.metadata => Ptr(Space::Generic(7), zero),
            _ => Value::number(0),
        }
    }
    fn shared_slot(&self, address: Value) -> Option<(usize, usize)> {
        let Ptr(Space::Shared(id), l) = address else {
            return None;
        };
        let byte = usize::try_from(l.value()?).ok()?;
        (byte % 2 == 0 && byte / 2 < self.shared[id].len()).then_some((id, byte / 2))
    }
    fn read(&mut self, op: &str, address: Value) -> Option<Value> {
        if op == "ld.shared.u16" {
            let (i, j) = self.shared_slot(address)?;
            self.reads[i][j] = true;
            return self.shared[i][j];
        }
        let Ptr(space, l) = address else { return None };
        if op == "ld.global.nc.u32" {
            return (self.metadata && space == Space::Metadata && l.value() == Some(8)).then_some(Value::number(1));
        }
        if l.tail != 0 || l.c < 0 || l.c % 2 != 0 {
            return None;
        }
        let i = l.c / 2;
        match space {
            Space::X if l.tile == 0 && i < self.k => Some(Half(Source::X, i as u16)),
            Space::W if l.tile == 2 * TILE as i64 * self.k && i < self.valid as i64 * self.k => {
                Some(Half(Source::W((i / self.k) as u8), (i % self.k) as u16))
            }
            _ => None,
        }
    }
    fn store(&mut self, op: &str, address: Value, v: Value) -> Option<()> {
        if op == "st.shared.u16" {
            let (i, j) = self.shared_slot(address)?;
            if self.writes[i][j].replace(v).is_some() {
                return None;
            }
            return Some(());
        }
        let Ptr(Space::Y, l) = address else { return None };
        let Dot(col, end) = v else { return None };
        if end as i64 != self.k
            || l.tile != 4 * TILE as i64
            || l.tail != 0
            || l.c != 4 * col as i64
            || col as usize >= self.valid
            || std::mem::replace(&mut self.outputs[col as usize], true)
        {
            return None;
        }
        Some(())
    }
}
struct Lane {
    regs: Vec<Value>,
    pc: usize,
    metadata_read: bool,
}
impl Lane {
    fn new(p: &Program<'_>, tid: i64) -> Self {
        Self {
            regs: p
                .names
                .iter()
                .map(|n| match *n {
                    "%tid.x" => Value::number(tid),
                    "%ctaid.x" => Value::number(0),
                    "%ctaid.y" => Int(Linear { c: 0, tile: 1, tail: 0 }),
                    _ => Unknown,
                })
                .collect(),
            pc: 0,
            metadata_read: false,
        }
    }
    fn value(&self, a: &Arg) -> Option<Value> {
        match a {
            Arg::Reg(r) => Some(self.regs[*r]),
            Arg::Imm(i) => Some(Value::number(*i)),
            Arg::Shared(i) => Some(Ptr(Space::Shared(*i), Linear::imm(0))),
            Arg::Mem(r, off) => self.regs[*r].add(Value::number(*off), 64),
            _ => None,
        }
    }
    fn run(&mut self, p: &Program<'_>, tile: &mut Tile) -> Option<bool> {
        for steps in 1..=12000 {
            let s = p.statements.get(self.pc)?;
            self.pc += 1;
            if let Some((r, yes)) = s.guard
                && self.regs[r].truth()? != yes
            {
                continue;
            }
            let a = &s.args;
            match s.op {
                "ret" | "bar.sync" => {
                    tile.remaining = tile.remaining.checked_sub(steps)?;
                    return Some(s.op == "bar.sync");
                }
                "bra" | "bra.uni" => {
                    let Arg::Label(to) = a[0] else { return None };
                    self.pc = to;
                    continue;
                }
                "st.shared.u16" | "st.global.f32" => {
                    tile.store(s.op, self.value(&a[0])?, self.value(&a[1])?)?;
                    continue;
                }
                _ => {}
            }
            let d = a[0].reg()?;
            let width = p.widths[d];
            let v = |i: usize| self.value(&a[i]);
            let c = |i| v(i)?.constant();
            let flag = |i| v(i)?.truth();
            let val = match s.op {
                "ld.param.u32" | "ld.param.u64" => {
                    let Arg::Param(i) = a[1] else { return None };
                    tile.param(i)
                }
                "ld.global.nc.u16" | "ld.shared.u16" => tile.read(s.op, v(1)?)?,
                "ld.global.nc.u32" => {
                    let val = tile.read(s.op, v(1)?)?;
                    if self.metadata_read {
                        return None;
                    }
                    self.metadata_read = true;
                    val
                }
                "mov.u32" | "mov.f32" | "cvt.s64.s32" => v(1)?,
                "cvta.to.global.u64" => match v(1)? {
                    Ptr(Space::Generic(i), off) => Ptr(
                        match i {
                            0 => Space::X,
                            2 => Space::W,
                            3 => Space::Y,
                            7 => Space::Metadata,
                            _ => return None,
                        },
                        off,
                    ),
                    _ => Unknown,
                },
                "cvt.u32.u64" => v(1)?.constant().map_or(Unknown, |i| Value::number(i as i32 as i64)),
                "add.s32" | "add.s64" => v(1)?.add(v(2)?, width)?,
                "mul.wide.s32" | "mul.lo.s64" => v(1)?.times(v(2)?, width)?,
                "mad.lo.s32" => v(1)?.times(v(2)?, width)?.add(v(3)?, width)?,
                "sub.s32" => v(1)?.add(v(2)?.times(Value::number(-1), 32)?, 32)?,
                "shl.b32" | "shl.b64" => {
                    let n = u32::try_from(c(2)?).ok()?;
                    if n >= width as u32 {
                        Value::number(0)
                    } else {
                        v(1)?.times(Value::number(1i64.checked_shl(n)?), width)?
                    }
                }
                "shr.s32" => Value::number(((c(1)? as i32) >> (c(2)? as u32).min(31)) as i64),
                "shr.u32" => Value::number((c(1)? as u32).checked_shr(c(2)? as u32).unwrap_or(0) as i32 as i64),
                "not.b32" => Value::number(!(c(1)? as i32) as i64),
                "and.b32" => Value::number(((c(1)? as i32) & (c(2)? as i32)) as i64),
                "max.s32" => Value::number(c(1)?.max(c(2)?)),
                "min.s32" => Value::number(c(1)?.min(c(2)?)),
                "setp.lt.u32" => Value::number(i64::from((c(1)? as u32) < (c(2)? as u32))),
                "setp.ne.s32" => Value::number(i64::from(!v(1)?.compare(v(2)?, "eq")?.truth()?)),
                "setp.eq.s64" | "setp.eq.s32" | "setp.lt.s32" | "setp.ge.s32" | "setp.gt.s32" => {
                    v(1)?.compare(v(2)?, s.op.split('.').nth(1)?)?
                }
                "and.pred" => Value::number(i64::from(flag(1)? && flag(2)?)),
                "or.pred" => Value::number(i64::from(flag(1)? || flag(2)?)),
                "cvt.rn.bf16.f32" if c(1) == Some(0) => Value::number(0),
                "mov.b32" => {
                    let Arg::Tuple(ts) = &a[1] else { return None };
                    match self.value(&ts[1])? {
                        Half(src, k) => Float(src, k),
                        Int(l) if l.value() == Some(0) => Value::number(0),
                        _ => Unknown,
                    }
                }
                "fma.rn.f32" => match (v(1)?, v(2)?, v(3)?) {
                    (Float(Source::X, k), Float(Source::W(col), k2), acc)
                        if k == k2 && (k == 0 && acc == Value::number(0) || acc == Dot(col, k)) =>
                    {
                        Dot(col, k + 1)
                    }
                    _ => Unknown,
                },
                _ => return None,
            };
            self.regs[d] = val;
        }
        None
    }
}
fn verify(p: &Program<'_>, k: i64, valid: usize, metadata: bool) -> bool {
    let mut tile = Tile::new(p, k, valid, metadata);
    let mut lanes: Vec<_> = (0..256).map(|tid| Lane::new(p, tid)).collect();
    for _ in 0..2 * k / 256 + 8 {
        let mut state = None;
        for lane in &mut lanes {
            let Some(active) = lane.run(p, &mut tile) else {
                return false;
            };
            let next = (active, if active { lane.pc } else { 0 });
            if state.is_some_and(|v| v != next) {
                return false;
            }
            state = Some(next);
        }
        if tile.barrier().is_none() {
            return false;
        }
        if state == Some((false, 0)) {
            return lanes.iter().all(|l| l.metadata_read == metadata) && tile.outputs[..valid].iter().all(|v| *v);
        }
    }
    false
}
pub(super) fn rewrite(signature: &str, text: &str) -> Option<String> {
    if text.len() > 100000 {
        return None;
    }
    let clean = crate::scan::blank_comments(text);
    let p = Program::parse_typed(signature, &clean, &TYPES, 256, true)?;
    // A backedge targeting the first instruction must never re-enter the
    // dispatch when the original body is running as the fallback.
    if p.labels.values().any(|pc| *pc == 0)
        || p.params.iter().any(|s| s.starts_with('%'))
        || p.shared.iter().any(|(s, _)| p.labels.contains_key(s))
        || p.shared.len() != 2
        || p.shared.iter().any(|(_, len)| *len < 8208)
        || p.widths.len() > 1024
        || p.statements.len() > 600
        || !typed(&p)
    {
        return None;
    }
    let depths: Vec<_> = DEPTHS
        .into_iter()
        .filter(|k| {
            [1, TILE]
                .into_iter()
                .all(|n| [false, true].into_iter().all(|m| verify(&p, *k, n, m)))
        })
        .collect();
    if depths.is_empty() {
        return None;
    }
    let pos = p.statements.first()?.position;
    let fast = emit::write(&p, &depths);
    Some(format!("{}{}{}", &text[..pos], fast, &text[pos..]))
}

#[cfg(test)]
mod tests;
