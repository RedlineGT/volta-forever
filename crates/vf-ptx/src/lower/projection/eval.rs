use super::{
    program::{Arg, Program, Register, Statement},
    symbolic::Symbolic,
    values::{Affine, Axis, Value, Word},
};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Bindings {
    pub rows: Register,
    pub counts: Register,
    pub expert: Register,
    pub first: Register,
    pub member: Register,
    pub row: Register,
    pub weights: Register,
}
pub(super) struct Frame {
    pub pc: usize,
    pub lane: usize,
    pub registers: Vec<Value>,
    pub binding: Option<Bindings>,
    pub guarded_indices: Option<(Register, Register, Register, Register, Option<Register>)>,
    pub epilogue_values: HashMap<Register, Word>,
    pub origins: Vec<(Register, usize)>,
    initial_count: Option<Register>,
    initial_rows: Option<Register>,
    changed: Vec<bool>,
    boundary: bool,
    epilogue: bool,
    tile_bound: bool,
    memory: HashMap<(bool, usize), u8>,
    store_offsets: HashSet<usize>,
    pub steps: usize,
}

impl Frame {
    pub fn new(p: &Program<'_>, lane: usize) -> Self {
        let mut registers = vec![Value::Unknown; p.names.len()];
        for (r, name) in p.names.iter().enumerate() {
            registers[r] = match *name {
                "%tid.x" => Value::Int(Affine::tid(lane)),
                "%ctaid.x" => Value::Int(Affine::axis(Axis::Cta)),
                "%nctaid.x" => Value::int(36),
                _ => Value::Unknown,
            }
        }
        Self {
            pc: 0,
            lane,
            registers,
            binding: None,
            guarded_indices: None,
            epilogue_values: HashMap::new(),
            origins: Vec::new(),
            initial_count: None,
            initial_rows: None,
            changed: vec![false; p.names.len()],
            boundary: false,
            epilogue: false,
            tile_bound: false,
            memory: HashMap::new(),
            store_offsets: HashSet::new(),
            steps: 0,
        }
    }
    pub fn val(&self, arg: &Arg) -> Value {
        match arg {
            Arg::Reg(r) => self.registers[*r],
            Arg::Imm(i) => Value::int(*i),
            _ => Value::Unknown,
        }
    }
    fn address(&self, arg: &Arg) -> Option<Affine> {
        let Arg::Mem(r, off) = arg else { return None };
        self.registers[*r].affine()?.plus(*off)
    }
    fn read(&mut self, arg: &Arg) -> Option<()> {
        if !self.epilogue {
            return Some(());
        }
        let mut ok = true;
        arg.visit(&mut |r| {
            if self.changed[r] {
                if let Some(value) = self.registers[r].bits() {
                    self.epilogue_values.entry(r).or_insert(value);
                } else {
                    ok = false;
                }
            }
        });
        ok.then_some(())
    }
    pub fn put(&mut self, r: Register, value: Value) {
        if self.initial_count == Some(r) {
            self.initial_count = None
        }
        if self.initial_rows == Some(r) {
            self.initial_rows = None
        }
        self.registers[r] = value;
        self.changed[r] = self.boundary && !self.epilogue;
    }
    fn symbol(&self, axis: Axis, p: &Program<'_>) -> Option<Register> {
        self.registers
            .iter()
            .enumerate()
            .find_map(|(r, &v)| (v == Value::Int(Affine::axis(axis)) && p.widths[r] == 32).then_some(r))
    }
    fn bind(&mut self, s: &Statement<'_>, p: &Program<'_>) -> Option<()> {
        if self.boundary || self.epilogue || !self.tile_bound {
            return None;
        }
        let Arg::Mem(weights, 0) = s.args.get(1)? else {
            return None;
        };
        if self.registers[*weights] != Value::Int(Affine::weights().plus(self.lane as i64 * 16)?) {
            return None;
        }
        let (count, rows) = (self.initial_count?, self.initial_rows?);
        let (e, f) = (self.symbol(Axis::Expert, p)?, self.symbol(Axis::First, p)?);
        let m = if self.lane < 4 {
            Some(self.symbol(Axis::Member, p)?)
        } else {
            None
        };
        self.guarded_indices = Some((count, rows, e, f, m));
        if self.lane == 0 {
            self.binding = Some(Bindings {
                rows: self.initial_rows?,
                counts: self.initial_count?,
                expert: self.symbol(Axis::Expert, p)?,
                first: self.symbol(Axis::First, p)?,
                member: self.symbol(Axis::Member, p)?,
                row: self.symbol(Axis::Row, p)?,
                weights: *weights,
            });
        }
        self.boundary = true;
        Some(())
    }
    fn load(&mut self, s: &Statement<'_>, math: &mut Symbolic) -> Option<()> {
        let [d, src] = s.args.as_slice() else { return None };
        let a = self.address(src)?;
        if self.epilogue {
            return None;
        }
        if s.op == "ld.global.nc.u32" {
            if self.boundary {
                return None;
            }
            let r = d.reg()?;
            let v = if a == Affine::axis(Axis::Count) {
                Affine::num(9)
            } else if a == Affine::item() {
                Affine::axis(Axis::Expert)
            } else if a == Affine::item().plus(4)? {
                Affine::axis(Axis::First)
            } else if a == Affine::item().plus(8)? {
                Affine::num(1)
            } else if a == Affine::member() {
                Affine::axis(Axis::Member)
            } else {
                return None;
            };
            // Establish the guards from their original noncoherent memory loads.
            // The address of an earlier guard cannot depend on a later field.
            let expected = [
                Affine::axis(Axis::Count),
                Affine::item(),
                Affine::item().plus(4)?,
                Affine::item().plus(8)?,
                Affine::member(),
            ];
            if expected.get(self.origins.len()) != Some(&a) {
                return None;
            }
            self.origins.push((r, self.pc - 1));
            self.put(r, Value::Int(v));
            if a == Affine::axis(Axis::Count) {
                self.initial_count = Some(r)
            }
            if a == Affine::item().plus(8)? {
                self.initial_rows = Some(r)
            }
            return Some(());
        }
        if !self.boundary || !matches!(s.op, "ld.global.nc.L1::no_allocate.v4.u32" | "ld.global.nc.v4.u32") {
            return None;
        }
        let Arg::Tuple(ds) = d else { return None };
        if ds.len() != 4 {
            return None;
        }
        let (w, off) = if let Some(offset) = a.offset(Affine::weights()) {
            (true, offset)
        } else {
            (false, a.offset(Affine::inputs())?)
        };
        let count = self.memory.entry((w, off)).or_default();
        *count = count.checked_add(1)?;
        for (i, d) in ds.iter().enumerate() {
            let at = off.checked_add(4 * i)?;
            let bits = if w { math.weight(at)? } else { math.input(at)? };
            self.put(d.reg()?, Value::Bits(bits));
        }
        Some(())
    }
    pub fn verified_memory(&self) -> bool {
        let mut wanted = HashMap::new();
        let l = self.lane;
        for g in 0..32 {
            for m in 0..2 {
                for v in [l, 32 + l, 64 + 2 * (l % 4), 65 + 2 * (l % 4)] {
                    *wanted.entry((true, 16 * (144 * g + 72 * m + v))).or_insert(0) += 1;
                }
            }
            if l < 4 {
                for v in [0, 16] {
                    wanted.insert((false, 128 * g + 32 * l + v), 1);
                }
            }
        }
        let stores: HashSet<_> = if l < 4 {
            (0..4).map(|v| 4 * l + 16 * v).collect()
        } else {
            HashSet::new()
        };
        wanted == self.memory && stores == self.store_offsets && self.boundary && self.epilogue
    }
    fn comparison(&mut self, op: &str, a: Affine, b: Affine, d: &Arg) -> Option<bool> {
        if op == "setp.gt.f32" {
            return Some(f32::from_bits(a.value()? as u32) > f32::from_bits(b.value()? as u32));
        }
        if let (Some(x), Some(y)) = (a.value(), b.value()) {
            let (x, y) = (x as i32, y as i32);
            return match op {
                "setp.ge.s32" => Some(x >= y),
                "setp.lt.s32" => Some(x < y),
                "setp.eq.s32" => Some(x == y),
                _ => None,
            };
        }
        if !self.tile_bound && op == "setp.ge.s32" && a == Affine::physical_unit() && b == Affine::num(144) {
            // Interpret the in-range hardware tile U as 16*(U/16) + U%16.
            // With the runtime grid and item-count guards every live warp owns
            // exactly one tile. Its only outer backedge cannot be taken.
            self.registers[d.reg()?] = Value::Int(Affine::unit());
            self.tile_bound = true;
            return Some(false);
        }
        // Before the guards, branch selection must remain independent of the
        // assumed upper bounds on expert, first member, and member indices.
        if !self.boundary {
            return None;
        }
        let d = a.signed()?.sub(b.signed()?)?;
        let (l, h) = d.range()?;
        match op {
            "setp.lt.s32" if l >= 0 => Some(false),
            "setp.lt.s32" if h < 0 => Some(true),
            "setp.ge.s32" if l >= 0 => Some(true),
            "setp.ge.s32" if h < 0 => Some(false),
            _ => None,
        }
    }
    fn int_op(&self, op: &str, a: Value, b: Value) -> Option<Affine> {
        let mut a = a.affine()?;
        let mut b = b.affine()?;
        if matches!(op, "add.s32" | "sub.s32" | "mul.wide.s32" | "mul.lo.s32" | "div.s32") {
            a = a.signed32()?;
            b = b.signed32()?;
        }
        if op == "shr.u32" {
            a = a.unsigned32()?;
        }

        let c = match op {
            "add.s32" | "add.s64" => a.add(b)?,
            "sub.s32" => a.sub(b)?,
            "mul.wide.s32" | "mul.lo.s32" | "mul.lo.s64" => {
                if op == "mul.wide.s32" {
                    a.signed()?;
                    b.signed()?;
                }
                if let Some(n) = b.value() {
                    a.times(n)?
                } else {
                    b.times(a.value()?)?
                }
            }
            "shl.b64" | "shl.b32" => {
                let shift = u32::try_from(b.value()?).ok()?;
                a.times(1i64.checked_shl(shift)?)?
            }
            "shr.u32" => a.shr_unsigned(b.value()?)?,
            "div.s32" if a == Affine::unit() && b == Affine::num(16) => Affine::axis(Axis::Item),
            "div.s32" if a == Affine::axis(Axis::Member) && b == Affine::num(9) => Affine::axis(Axis::Row),
            "div.s32" => Affine::num(i64::from((a.value()? as i32).checked_div(b.value()? as i32)?)),
            "and.b32" => a.mask(b.value()?)?,
            "or.b32" => a.or_literal(b.value()?)?,
            _ => return None,
        };
        if matches!(op, "add.s32" | "sub.s32" | "mul.lo.s32") {
            a.signed()?;
            b.signed()?;
            c.signed()
        } else if op == "shl.b32" && c.value().is_none() {
            c.signed()
        } else {
            Some(c)
        }
    }
    pub fn advance(&mut self, p: &Program<'_>, first: usize, end: usize, math: &mut Symbolic) -> Option<Option<usize>> {
        loop {
            let pc = self.pc;
            let s = p.statements.get(pc)?;
            self.steps += 1;
            if self.steps > 100_000 {
                return None;
            }
            if pc == first {
                self.bind(s, p)?;
            }
            if pc == end {
                if !self.boundary || self.epilogue {
                    return None;
                }
                self.epilogue = true;
            }
            self.pc += 1;
            if let Some((r, on)) = s.guard {
                self.read(&Arg::Reg(r))?;
                let test = self.registers[r].scalar()?;
                if !matches!(test, 0 | 1) {
                    return None;
                }
                if (test != 0) != on {
                    continue;
                }
            }
            if s.op == "ret" {
                return Some(None);
            }
            if let [Arg::Label(to)] = s.args.as_slice() {
                if !matches!(s.op, "bra" | "bra.uni") {
                    return None;
                }
                self.pc = *to;
                continue;
            }
            let (d, src) = s.args.split_first()?;
            for a in src {
                self.read(a)?;
            }
            if s.op == "st.global.u32" {
                self.read(d)?;
                let off = self.address(d)?.offset(Affine::output())?;
                if !self.epilogue || !self.store_offsets.insert(off) {
                    return None;
                }
                continue;
            }
            if matches!(
                s.op,
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32" | "shfl.sync.bfly.b32"
            ) {
                if !self.boundary || self.epilogue {
                    return None;
                }
                return Some(Some(pc));
            }
            if s.op.starts_with("ld.global") {
                self.load(s, math)?;
                continue;
            }
            let r = d.reg()?;
            let a = src.first().map_or(Value::Unknown, |v| self.val(v));
            let b = src.get(1).map_or(Value::Unknown, |v| self.val(v));
            let c = src.get(2).map_or(Value::Unknown, |v| self.val(v));
            let result = match s.op {
                "ld.param.u32" | "ld.param.u64" | "ld.param.f32" => {
                    let [Arg::Param(i)] = src else { return None };
                    let valid = match i {
                        0 | 3 | 6 | 7 | 8 | 9 => s.op == "ld.param.u64" && p.widths[r] == 64,
                        11 => s.op == "ld.param.f32" && p.widths[r] == 32,
                        _ => s.op == "ld.param.u32" && p.widths[r] == 32,
                    };
                    if !valid {
                        return None;
                    }
                    Value::Int(Affine::param(*i)?)
                }
                "cvta.to.global.u64" | "mov.u32" | "mov.b32" | "mov.f32" if src.len() == 1 => a,
                "cvt.s64.s32" => Value::Int(a.affine()?.signed32()?),
                "cvt.u64.u32" => Value::Int(a.affine()?.unsigned32()?),
                "setp.ge.s32" | "setp.eq.s32" | "setp.lt.s32" | "setp.gt.f32" => {
                    Value::int(self.comparison(s.op, a.affine()?, b.affine()?, src.first()?)? as i64)
                }
                "add.s32" | "add.s64" | "sub.s32" | "mul.wide.s32" | "mul.lo.s32" | "mul.lo.s64" | "div.s32"
                | "shl.b64" | "shr.u32" | "shl.b32" | "and.b32" | "or.b32" => {
                    if let Some(x) = self.int_op(s.op, a, b) {
                        Value::Int(x)
                    } else {
                        let x = match s.op {
                            "shr.u32" | "shl.b32" => a
                                .bits()
                                .zip(b.scalar())
                                .and_then(|(w, k)| w.shift(k, s.op == "shl.b32")),
                            "and.b32" | "or.b32" => a.bits().zip(b.bits()).and_then(|(x, y)| x.binary(y, s.op)),
                            _ => None,
                        };
                        x.map_or(Value::Unknown, Value::Bits)
                    }
                }
                "fma.rn.bf16x2" => a
                    .bits()
                    .zip(b.bits())
                    .zip(c.bits())
                    .and_then(|((x, y), z)| math.q4(x, y, z))
                    .map_or(Value::Unknown, Value::Bits),
                "fma.rn.f32" => a
                    .bits()
                    .zip(b.bits())
                    .zip(c.bits())
                    .and_then(|((x, y), z)| math.fma(x, y, z))
                    .map_or(Value::Unknown, Value::Bits),
                "add.f32" | "add.rn.f32" => a
                    .bits()
                    .zip(b.bits())
                    .and_then(|(x, y)| math.add(x, y))
                    .map_or(Value::Unknown, Value::Bits),
                "neg.f32" | "min.f32" | "max.f32" | "cvt.sat.f32.f32" | "ex2.approx.ftz.f32" | "div.rn.f32"
                | "mul.f32" | "cvt.rn.bf16.f32" | "cvt.rn.bf16x2.f32" | "fma.rm.f32"
                    if self.epilogue =>
                {
                    Value::Unknown
                }
                _ => return None,
            };
            // Fixed-width integer operations write a bit pattern. In particular a
            // left shift may discard the high bit before an unsigned right shift.
            let result = match result {
                Value::Int(v) if p.widths[r] == 32 && v.value().is_some() => Value::int((v.value()? as u32).into()),
                other => other,
            };
            self.put(r, result);
        }
    }
}
