//! Track addresses as affine expressions in opaque GPU addresses and the tile's
//! logical input indices. All 32-bit signed calculations must be overflow free.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(usize)]
pub(super) enum Axis {
    A,
    W,
    Items,
    Count,
    Members,
    Out,
    Expert,
    Item,
    First,
    Member,
    Row,
    Tile,
    Warp,
    Cta,
}
use Axis::*;
const DIM: usize = 14;
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub(super) struct Affine {
    pub k: i64,
    pub c: [i64; DIM],
}
impl Affine {
    pub fn num(k: i64) -> Self {
        Self { k, ..Self::default() }
    }
    pub fn axis(a: Axis) -> Self {
        let mut v = Self::default();
        v.c[a as usize] = 1;
        v
    }
    pub fn value(self) -> Option<i64> {
        self.c.iter().all(|&x| x == 0).then_some(self.k)
    }
    pub fn add(self, b: Self) -> Option<Self> {
        let mut v = Self::num(self.k.checked_add(b.k)?);
        for (i, c) in v.c.iter_mut().enumerate() {
            *c = self.c[i].checked_add(b.c[i])?;
        }
        Some(v)
    }
    pub fn times(self, k: i64) -> Option<Self> {
        let mut v = Self::num(self.k.checked_mul(k)?);
        for (to, from) in v.c.iter_mut().zip(self.c) {
            *to = from.checked_mul(k)?;
        }
        Some(v)
    }
    pub fn sub(self, b: Self) -> Option<Self> {
        self.add(b.times(-1)?)
    }
    pub fn plus(self, k: i64) -> Option<Self> {
        self.add(Self::num(k))
    }
    pub fn range(self) -> Option<(i64, i64)> {
        if self.c[..6].iter().any(|&v| v != 0) {
            return None;
        }
        // These limits must be guarded before entering the generated path.
        let limits = [65535, 8, 1048575, 1048575, 1048575, 15, 3, 35];
        let (mut lo, mut hi) = (self.k, self.k);
        for (coef, max) in self.c[6..].iter().zip(limits) {
            let d = coef.checked_mul(max)?;
            lo = lo.checked_add(d.min(0))?;
            hi = hi.checked_add(d.max(0))?;
        }
        Some((lo, hi))
    }
    pub fn signed(self) -> Option<Self> {
        let (l, h) = self.range()?;
        (l >= i32::MIN.into() && h <= i32::MAX.into()).then_some(self)
    }
    pub fn signed32(self) -> Option<Self> {
        match self.value() {
            Some(x) => Some(Self::num((x as u32 as i32).into())),
            None => self.signed(),
        }
    }
    pub fn unsigned32(self) -> Option<Self> {
        if let Some(x) = self.value() {
            return Some(Self::num((x as u32).into()));
        }
        let (lo, hi) = self.range()?;
        (lo >= 0 && hi <= u32::MAX.into()).then_some(self)
    }
    pub fn mask(self, bits: i64) -> Option<Self> {
        if let Some(n) = self.value() {
            return Some(Self::num(n & bits));
        }
        if bits < 0 || bits > i64::from(u32::MAX) {
            return None;
        }
        let top = 64 - bits.leading_zeros();
        self.c
            .iter()
            .all(|c| (c & ((1i64 << top) - 1)) == 0)
            .then(|| Self::num(self.k & bits))
    }
    pub fn or_literal(self, bits: i64) -> Option<Self> {
        if let Some(n) = self.value() {
            return Some(Self::num(n | bits));
        }
        if self.mask(bits)?.value()? == 0 {
            self.plus(bits)
        } else {
            None
        }
    }
    pub fn shr_unsigned(self, s: i64) -> Option<Self> {
        if !(0..32).contains(&s) || self.range()?.0 < 0 {
            return None;
        }
        let d = 1 << s;
        let mut v = Self::num(self.k >> s);
        for (i, c) in self.c.iter().enumerate() {
            if c % d != 0 {
                return None;
            }
            v.c[i] = c / d;
        }
        Some(v)
    }
    pub fn tid(lane: usize) -> Self {
        Self::axis(Warp).times(32).unwrap().plus(lane as i64).unwrap()
    }
    pub fn unit() -> Self {
        Self::axis(Item).times(16).unwrap().add(Self::axis(Tile)).unwrap()
    }
    pub fn physical_unit() -> Self {
        Self::axis(Cta).times(4).unwrap().add(Self::axis(Warp)).unwrap()
    }
    pub fn param(i: usize) -> Option<Self> {
        Some(match i {
            0 => Self::axis(A),
            1 => Self::num(2048),
            2 => Self::num(9),
            3 => Self::axis(W),
            4 => Self::num(32),
            5 => Self::num(16),
            6 => Self::axis(Items),
            7 => Self::axis(Count),
            8 => Self::axis(Members),
            9 => Self::axis(Out),
            10 => Self::num(512),
            11 => Self::num(0),
            _ => return None,
        })
    }
    pub fn item() -> Self {
        Self::axis(Items).add(Self::axis(Item).times(12).unwrap()).unwrap()
    }
    pub fn member() -> Self {
        Self::axis(Members).add(Self::axis(First).times(4).unwrap()).unwrap()
    }
    pub fn weights() -> Self {
        Self::axis(W)
            .add(Self::axis(Expert).times(16 * 32 * 2304).unwrap())
            .unwrap()
            .add(Self::axis(Tile).times(32 * 2304).unwrap())
            .unwrap()
    }
    pub fn inputs() -> Self {
        Self::axis(A).add(Self::axis(Row).times(4096).unwrap()).unwrap()
    }
    pub fn output() -> Self {
        Self::axis(Out)
            .add(Self::axis(Member).times(1024).unwrap())
            .unwrap()
            .add(Self::axis(Tile).times(64).unwrap())
            .unwrap()
    }
    pub fn offset(self, base: Self) -> Option<usize> {
        (self.c == base.c)
            .then(|| usize::try_from(self.k.checked_sub(base.k)?).ok())
            .flatten()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(super) struct Word(pub [u32; 32]);
impl Word {
    pub fn literal(v: u32) -> Self {
        Self(std::array::from_fn(|i| (v >> i) & 1))
    }
    pub fn literal_bits(self) -> Option<u32> {
        self.0
            .iter()
            .enumerate()
            .try_fold(0, |a, (i, &b)| (b < 2).then(|| a | (b << i)))
    }
    pub fn tag(self, offset: usize, len: usize) -> Option<u32> {
        let v = *self.0.get(offset)?;
        if v < 32 || !v.is_multiple_of(32) {
            return None;
        }
        self.0
            .get(offset..offset + len)?
            .iter()
            .enumerate()
            .all(|(i, &bit)| bit == v + i as u32)
            .then_some(v / 32)
    }
    pub fn binary(self, b: Self, op: &str) -> Option<Self> {
        let mut d = [0; 32];
        for (i, j) in d.iter_mut().enumerate() {
            let (x, y) = (self.0[i], b.0[i]);
            *j = match op {
                "and.b32" if x == 0 || y == 0 => 0,
                "and.b32" if x == 1 || x == y => y,
                "and.b32" if y == 1 => x,
                "or.b32" if x == 1 || y == 1 => 1,
                "or.b32" if x == 0 || x == y => y,
                "or.b32" if y == 0 => x,
                _ => return None,
            };
        }
        Some(Self(d))
    }
    pub fn shift(self, x: i64, left: bool) -> Option<Self> {
        let k = usize::try_from(x).ok()?.min(32);
        Some(Self(std::array::from_fn(|i| {
            if left {
                i.checked_sub(k).map_or(0, |j| self.0[j])
            } else {
                self.0.get(i + k).copied().unwrap_or(0)
            }
        })))
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Value {
    Unknown,
    Int(Affine),
    Bits(Word),
}
impl Value {
    pub fn int(v: i64) -> Self {
        Self::Int(Affine::num(v))
    }
    pub fn affine(self) -> Option<Affine> {
        match self {
            Self::Int(v) => Some(v),
            Self::Bits(v) => Some(Affine::num(v.literal_bits()?.into())),
            Self::Unknown => None,
        }
    }
    pub fn bits(self) -> Option<Word> {
        match self {
            Self::Bits(w) => Some(w),
            Self::Int(v) => Some(Word::literal(v.value()? as u32)),
            Self::Unknown => None,
        }
    }
    pub fn scalar(self) -> Option<i64> {
        self.affine()?.value()
    }
}
