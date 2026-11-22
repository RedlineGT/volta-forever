//! Integer expressions are affine in the current column tile and in the number
//! of extra columns past a full tile. Unknown values may flow through unused
//! arithmetic, but can never choose a branch, an address or an output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Linear {
    pub c: i64,
    pub tile: i64,
    pub tail: i64,
}
impl Linear {
    pub const fn imm(c: i64) -> Self {
        Self { c, tile: 0, tail: 0 }
    }
    pub fn value(self) -> Option<i64> {
        (self.tile == 0 && self.tail == 0).then_some(self.c)
    }
    pub fn plus(self, v: Self) -> Option<Self> {
        Some(Self {
            c: self.c.checked_add(v.c)?,
            tile: self.tile.checked_add(v.tile)?,
            tail: self.tail.checked_add(v.tail)?,
        })
    }
    pub fn scale(self, n: i64) -> Option<Self> {
        Some(Self {
            c: self.c.checked_mul(n)?,
            tile: self.tile.checked_mul(n)?,
            tail: self.tail.checked_mul(n)?,
        })
    }
    pub fn range(self) -> Option<(i64, i64)> {
        let b = self.tile.checked_mul(4095)?;
        let n = self.tail.checked_mul(65535)?;
        Some((
            self.c.checked_add(b.min(0))?.checked_add(n.min(0))?,
            self.c.checked_add(b.max(0))?.checked_add(n.max(0))?,
        ))
    }
    pub fn signed(self, w: u8) -> Option<Self> {
        if let Some(i) = self.value() {
            return Some(Self::imm(if w == 32 { i as i32 as i64 } else { i }));
        }
        let (lo, hi) = self.range()?;
        (w == 64 || lo >= i32::MIN as i64 && hi <= i32::MAX as i64).then_some(self)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Space {
    X,
    W,
    Y,
    Metadata,
    Generic(u8),
    Shared(usize),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Source {
    X,
    W(u8),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Value {
    Unknown,
    Int(Linear),
    Ptr(Space, Linear),
    Half(Source, u16),
    Float(Source, u16),
    Dot(u8, u16),
}
use Value::*;
impl Value {
    pub fn number(x: i64) -> Self {
        Int(Linear::imm(x))
    }
    pub fn constant(self) -> Option<i64> {
        if let Int(a) = self { a.value() } else { None }
    }
    pub fn truth(self) -> Option<bool> {
        match self.constant()? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
    pub fn add(self, b: Self, w: u8) -> Option<Self> {
        Some(match (self, b) {
            (Int(a), Int(b)) => Int(a.plus(b)?.signed(w)?),
            (Ptr(p, a), Int(b)) | (Int(b), Ptr(p, a)) => Ptr(p, a.plus(b)?.signed(w)?),
            _ => Unknown,
        })
    }
    pub fn times(self, b: Self, w: u8) -> Option<Self> {
        if self.constant() == Some(0) || b.constant() == Some(0) {
            return Some(Self::number(0));
        }
        Some(match (self, b) {
            (Int(x), Int(y)) => {
                let part = if let Some(c) = x.value() {
                    y.scale(c)
                } else {
                    x.scale(y.value()?)
                };
                Int(part?.signed(w)?)
            }
            _ => Unknown,
        })
    }
    pub fn compare(self, b: Self, op: &str) -> Option<Self> {
        let yes = match (self, b) {
            (Ptr(space, x), Int(y)) | (Int(y), Ptr(space, x))
                if !matches!(space, Space::Shared(_)) && x.value() == Some(0) && y.value() == Some(0) && op == "eq" =>
            {
                false
            }
            (Int(a), Int(b)) => {
                let (lo, hi) = a.plus(b.scale(-1)?)?.range()?;
                match op {
                    "eq" if lo == 0 && hi == 0 => true,
                    "eq" if lo > 0 || hi < 0 => false,
                    "lt" if hi < 0 => true,
                    "lt" if lo >= 0 => false,
                    "ge" if lo >= 0 => true,
                    "ge" if hi < 0 => false,
                    "gt" if lo > 0 => true,
                    "gt" if hi <= 0 => false,
                    _ => return None,
                }
            }
            _ => return None,
        };
        Some(Self::number(i64::from(yes)))
    }
}
