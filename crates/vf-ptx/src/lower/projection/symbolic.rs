//! Bit origins of packed weights and activations, and exact expression identities for
//! the floating-point dependencies. A tag is interned, never compared by its hash.
use super::values::Word;
use std::collections::HashMap;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Term {
    Input(u16),
    Raw {
        group: u8,
        table: u8,
        column: u8,
        k: u8,
    },
    Nibble {
        group: u8,
        table: u8,
        column: u8,
        k: u8,
    },
    Field {
        group: u8,
        table: u8,
        column: u8,
        bias: bool,
    },
    Dot {
        group: u8,
        table: u8,
        column: u8,
        part: u8,
    },
    Bits(Word),
    Add(u32, u32),
    Fma(u32, u32, u32),
}
#[derive(Default)]
pub(super) struct Symbolic {
    terms: Vec<Term>,
    ids: HashMap<Term, u32>,
}
impl Symbolic {
    fn id(&mut self, t: Term) -> Option<u32> {
        if let Some(&id) = self.ids.get(&t) {
            return Some(id);
        }
        if self.terms.len() >= 400_000 {
            return None;
        }
        let id = self.terms.len() as u32 + 1;
        self.terms.push(t);
        self.ids.insert(t, id);
        Some(id)
    }
    fn put(&mut self, w: &mut Word, at: usize, len: usize, t: Term) -> Option<()> {
        let start = self.id(t)? * 32;
        for (i, b) in w.0.get_mut(at..at + len)?.iter_mut().enumerate() {
            *b = start + i as u32;
        }
        Some(())
    }
    pub fn word(&mut self, t: Term) -> Option<Word> {
        let mut w = Word::literal(0);
        self.put(&mut w, 0, 32, t)?;
        Some(w)
    }
    pub fn term(&self, w: Word, at: usize, count: usize) -> Option<Term> {
        self.terms.get(w.tag(at, count)? as usize - 1).copied()
    }
    pub fn input(&mut self, offset: usize) -> Option<Word> {
        if !offset.is_multiple_of(4) || offset + 4 > 4096 {
            return None;
        }
        let mut w = Word::literal(0);
        for i in 0..2 {
            self.put(&mut w, i * 16, 16, Term::Input((offset / 2 + i) as u16))?
        }
        Some(w)
    }
    pub fn weight(&mut self, offset: usize) -> Option<Word> {
        if !offset.is_multiple_of(4) || offset >= 32 * 2304 {
            return None;
        }
        let g = (offset / 2304) as u8;
        let m = ((offset / 1152) % 2) as u8;
        let i = (offset / 4) % 288;
        let mut w = Word::literal(0);
        if i < 256 {
            let f = (i / 128) * 128 + (i % 4) * 32 + (i % 128) / 4;
            let column = (f / 64) * 8 + f / 4 % 8;
            let k0 = f % 4 * 16 + f / 32 % 2 * 8;
            for h in 0..2 {
                for n in 0..4 {
                    self.put(
                        &mut w,
                        h * 16 + n * 4,
                        4,
                        Term::Raw {
                            group: g,
                            table: m,
                            column: column as u8,
                            k: (k0 + 2 * n + h) as u8,
                        },
                    )?;
                }
            }
        } else {
            let s = i - 256;
            let col = (s % 4) * 8 + (s / 8) * 2;
            let b = s / 4 % 2 == 1;
            for h in 0..2 {
                self.put(
                    &mut w,
                    h * 16,
                    16,
                    Term::Field {
                        group: g,
                        table: m,
                        column: (col + h) as u8,
                        bias: b,
                    },
                )?;
            }
        }
        Some(w)
    }
    pub fn q4(&mut self, a: Word, b: Word, c: Word) -> Option<Word> {
        let coeff = (a.literal_bits()?, b.literal_bits()?);
        if !matches!(coeff, (0x43004300, 0xbf80bf80) | (0xbf80bf80, 0x43004300)) {
            return None;
        }
        let mut w = Word::literal(0);
        let mask = Word::literal(0x43004300);
        for at in [0, 16] {
            if c.0[at + 4..at + 16] != mask.0[at + 4..at + 16] {
                return None;
            }
            let Term::Raw {
                group,
                table,
                column,
                k,
            } = self.term(c, at, 4)?
            else {
                return None;
            };
            self.put(
                &mut w,
                at,
                16,
                Term::Nibble {
                    group,
                    table,
                    column,
                    k,
                },
            )?;
        }
        Some(w)
    }
    pub fn add(&mut self, a: Word, b: Word) -> Option<Word> {
        if a == Word::literal(0) && b == a {
            return Some(a);
        }
        let x = self.id(Term::Bits(a))?;
        let y = self.id(Term::Bits(b))?;
        self.word(Term::Add(x, y))
    }
    pub fn fma(&mut self, a: Word, b: Word, c: Word) -> Option<Word> {
        let x = self.id(Term::Bits(a))?;
        let y = self.id(Term::Bits(b))?;
        let z = self.id(Term::Bits(c))?;
        self.word(Term::Fma(x, y, z))
    }
    fn bf32(&mut self, t: Term) -> Option<Word> {
        let mut w = Word::literal(0);
        self.put(&mut w, 16, 16, t)?;
        Some(w)
    }
    fn xsum(&mut self, group: u8, lane: u8) -> Option<Word> {
        let mut v = [Word::literal(0); 4];
        for (q, part) in v.iter_mut().enumerate() {
            for i in 0..16 {
                let x = self.bf32(Term::Input(u16::from(group) * 64 + (16 * q + i) as u16))?;
                *part = if i == 0 {
                    self.add(x, *part)?
                } else {
                    self.add(*part, x)?
                };
            }
        }
        let l = usize::from(lane % 4);
        let a = self.add(v[l], v[l ^ 1])?;
        let b = self.add(v[l ^ 2], v[l ^ 3])?;
        self.add(a, b)
    }
    pub fn expected_outputs(&mut self, lane: u8) -> Option<HashMap<Word, (u8, u8)>> {
        let mut out = HashMap::new();
        for table in 0..2u8 {
            for j in 0..4u8 {
                for h in 0..2u8 {
                    let column = j * 8 + (lane % 4) * 2 + h;
                    let mut acc = Word::literal(0);
                    for group in 0..32u8 {
                        let dot = self.word(Term::Dot {
                            group,
                            table,
                            column,
                            part: 12,
                        })?;
                        let sc = self.bf32(Term::Field {
                            group,
                            table,
                            column,
                            bias: false,
                        })?;
                        let bi = self.bf32(Term::Field {
                            group,
                            table,
                            column,
                            bias: true,
                        })?;
                        let x = self.xsum(group, lane)?;
                        let tmp = self.fma(dot, sc, acc)?;
                        acc = self.fma(x, bi, tmp)?;
                    }
                    if out.insert(acc, (table, column)).is_some() {
                        return None;
                    }
                }
            }
        }
        Some(out)
    }
}
