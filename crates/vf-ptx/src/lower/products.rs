//! FP16 tensor-core products and the chain arrangement: `tanh` expansion, the two-instruction
//! Volta scheme for FP16 products (even/odd k and rows/columns), f16 and f32 accumulators,
//! and the single-instruction chain form for runs of products on the same accumulator.

use std::fmt::Write as _;

use crate::scan;

use super::body::Body;
use super::lane::Lane;
use std::collections::HashSet;

use super::types::{Form, first_operand};
use super::{TANH, Unsupported};

impl<'a> Body<'a> {
    /// `tanh.approx.f32 d, s;` as instructions every target has: tanh(s) = 1 - 2 / (exp(2s) + 1), and
    /// s itself where |s| < 0.004 (the formula cancels there; the error of returning s is below
    /// s^3 / 3). Saturates to +-1 like the instruction.
    pub(super) fn tanh(&mut self, operands: &str) -> Result<(), Unsupported> {
        let (dst, src) = match scan::split_operands(operands)[..] {
            [dst, src] if scan::register(dst).is_some() => (dst, src),
            _ => return Err(Unsupported(format!("{TANH}{operands}: operands not recognised"))),
        };
        emit!(self, "{{");
        emit!(self, ".reg .f32 %vf_e, %vf_a;");
        emit!(self, ".reg .pred %vf_p;");
        emit!(self, "mul.f32 %vf_e, {src}, 0f4038AA3B;");
        emit!(self, "ex2.approx.f32 %vf_e, %vf_e;");
        emit!(self, "add.f32 %vf_e, %vf_e, 0f3F800000;");
        emit!(self, "rcp.approx.f32 %vf_e, %vf_e;");
        emit!(self, "fma.rn.f32 %vf_e, %vf_e, 0fC0000000, 0f3F800000;");
        emit!(self, "abs.f32 %vf_a, {src};");
        emit!(self, "setp.lt.f32 %vf_p, %vf_a, 0f3B83126F;");
        emit!(self, "selp.f32 {dst}, {src}, %vf_e, %vf_p;");
        self.out.push_str("\n}");
        self.written(dst);
        self.counts.tanh += 1;
        Ok(())
    }

    /// The A registers of an FP16 mma with their halves swapped.
    pub(super) fn swapped(&mut self, a: &[&str]) -> Vec<String> {
        if let Some(found) = self.recall(Form::Swapped, a) {
            return found;
        }
        let x: [String; 2] = self.fresh_n();
        for (x, a) in x.iter().zip(a) {
            emit!(self, "shf.l.wrap.b32 {x}, {a}, {a}, 16;");
        }
        self.remember(Form::Swapped, a, &x);
        x.to_vec()
    }

    /// The B operands of the two Volta instructions for the FP16 B register `b`: registers 0 and 1 for
    /// the first one, 2 and 3 for the second.
    pub(super) fn columns(&mut self, b: &str) -> Vec<String> {
        if self.plan.columns.contains(b) {
            return self.operand(Form::Columns, &[b], false);
        }
        if let Some(found) = self.recall(Form::Columns, &[b]) {
            return found;
        }
        // In the lanes of an even group c: the elements (2k, c), (2k, c+1) and (2k+1, c), (2k+1, c+1)
        // of B, where k is the thread. (The lanes of odd groups compute values nobody reads.)
        let next = self.exchange(b, 4);
        let [low, even, high, odd] = self.fresh_n();
        emit!(self, "shl.b32 {low}, {b}, 16;");
        emit!(self, "shf.r.wrap.b32 {even}, {low}, {next}, 16;");
        emit!(self, "shr.u32 {high}, {next}, 16;");
        emit!(self, "shf.l.wrap.b32 {odd}, {b}, {high}, 16;");
        // The upper half of the warp reads groups 2, 3, 6, 7 and wants the other parity.
        let [first, second] = self.fresh_n();
        let upper = self.lane(Lane::Has(8));
        emit!(self, "selp.b32 {first}, {odd}, {even}, {upper};");
        emit!(self, "selp.b32 {second}, {even}, {odd}, {upper};");
        let y = vec![
            self.shuffle(&first, Lane::Columns(false)),
            self.shuffle(&first, Lane::Columns(true)),
            self.shuffle(&second, Lane::Columns(false)),
            self.shuffle(&second, Lane::Columns(true)),
        ];
        self.remember(Form::Columns, &[b], &y);
        y
    }

    /// The two Volta instructions of an FP16 mma: accumulators `c` in, the result in fresh registers.
    pub(super) fn volta_pair<const N: usize>(&mut self, types: &str, a: &[&str], b: &str, c: &[String]) -> [String; N] {
        // The A and B operands of the first instruction (registers 0 and 1) and of the second.
        let (layouts, x, y) = if self.plan.in_halves(a, b) {
            // k 0..3, then k 4..7: rows of A, columns of B, as their `ldmatrix` loaded them.
            let x = self.operand(Form::Rows, a, false);
            ("row.col", x, self.operand(Form::Cols, &[b], false))
        } else {
            // Even k, then odd k.
            let mut x = vec![a[0].to_string(), a[1].to_string()];
            x.extend(self.swapped(a));
            ("col.row", x, self.columns(b))
        };
        let [middle, result]: [[String; N]; 2] = [self.fresh_n(), self.fresh_n()];
        let opcode = format!("mma.sync.aligned.m8n8k4.{layouts}.{types}");
        emit!(
            self,
            "{opcode} {{{}}}, {{{}, {}}}, {{{}, {}}}, {{{}}};",
            middle.join(", "),
            x[0],
            x[1],
            y[0],
            y[1],
            c.join(", ")
        );
        emit!(
            self,
            "{opcode} {{{}}}, {{{}, {}}}, {{{}, {}}}, {{{}}};",
            result.join(", "),
            x[2],
            x[3],
            y[2],
            y[3],
            middle.join(", ")
        );
        result
    }

    /// The two products of a pair (`Plan::pairs`), at the second: for each `m16n8k8`, two of Volta's
    /// products on the rows of A and the columns of B of the pair, k 0..3 then k 4..7, in place on
    /// the eight accumulator registers of the program (`docs/pairs.md`, sections 2 and 13).
    pub(super) fn mma_pair(&mut self) {
        let Some(pair) = self.plan.pairs.second.get(&self.current_index) else {
            return;
        };
        let accumulators = pair.x.iter().chain(&pair.y).copied().collect::<Vec<&str>>().join(", ");
        let operands: Vec<([&str; 2], [&str; 2])> = pair
            .a
            .chunks(2)
            .zip(pair.bx.iter().zip(&pair.by))
            .map(|(a, (bx, by))| ([a[0], a[1]], [*bx, *by]))
            .collect();
        let written: Vec<&str> = pair.x.iter().chain(&pair.y).copied().collect();
        for (a, b) in operands {
            let rows = self.operand(Form::QRows, &a, false);
            let cols = self.operand(Form::QCols, &b, false);
            for half in 0..2 {
                emit!(
                    self,
                    "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {{{accumulators}}}, {{{}, {}}}, {{{}, {}}}, {{{accumulators}}};",
                    rows[2 * half],
                    rows[2 * half + 1],
                    cols[2 * half],
                    cols[2 * half + 1]
                );
            }
        }
        for name in written {
            self.written(name);
        }
    }

    /// `mma.m16n8k8` with f32 accumulators, or `mma.m16n8k16` as two of them in a row (`a` and `b`
    /// hold the operands of one after the other).
    pub(super) fn mma_f32(&mut self, d: &[&str], a: &[&str], b: &[&str], c: &[&str]) {
        let mut result: Vec<String> = self.recall(Form::Wide, c).unwrap_or_else(|| {
            // Registers 0..3 are read by even threads and 4..7 by odd ones.
            c.iter().chain(c).map(|name| name.to_string()).collect()
        });
        for (a, b) in a.chunks(2).zip(b) {
            result = self.volta_pair::<8>("f32.f16.f16.f32", a, b, &result).to_vec();
        }
        let odd = self.lane(Lane::Has(1));
        for name in d {
            self.written(name);
        }
        for (at, name) in d.iter().enumerate() {
            if *name != "_" {
                emit!(self, "selp.b32 {name}, {}, {}, {odd};", result[4 + at], result[at]);
            }
        }
        if !d.contains(&"_") {
            self.remember(Form::Wide, d, &result);
        }
    }

    /// `mma.m16n8k8` with f16 accumulators.
    ///
    /// In Volta layout the thread `t` of a group holds, in registers `2 * (t & 1)` and
    /// `2 * (t & 1) + 1`, what the Turing layout puts in register `t >> 1` of the threads `t & 1` and
    /// `(t & 1) + 2` of the group: one register of its own and one of the thread two lanes away.
    pub(super) fn mma_f16(&mut self, d: &[&str], a: &[&str], b: &[&str], c: &[&str]) {
        let high = self.lane(Lane::Has(2));
        let mut result = self.recall(Form::Half, c).unwrap_or_else(|| {
            let [mine, low, upper]: [String; 3] = self.fresh_n();
            emit!(self, "selp.b32 {mine}, {}, {}, {high};", c[0], c[1]);
            let theirs = self.exchange(&mine, 2);
            emit!(self, "selp.b32 {low}, {theirs}, {}, {high};", c[0]);
            emit!(self, "selp.b32 {upper}, {}, {theirs}, {high};", c[1]);
            vec![low.clone(), upper.clone(), low, upper]
        });
        for (a, b) in a.chunks(2).zip(b) {
            result = self.volta_pair::<4>("f16.f16.f16.f16", a, b, &result).to_vec();
        }
        // The registers of the program are not written here: `half_end_one` gives them the
        // result where something may read them. What was derived from them is no longer theirs.
        for name in d {
            self.written(name);
        }
        self.remember(Form::Half, d, &result);
        self.kept_halves.insert([d[0].to_string(), d[1].to_string()], result);
    }

    /// Give an f16 accumulator kept in Volta layout back to the two registers of the program.
    pub(super) fn half_end_one(&mut self, key: [String; 2]) {
        let Some(result) = self.kept_halves.remove(&key) else {
            return;
        };
        let (high, odd) = (self.lane(Lane::Has(2)), self.lane(Lane::Has(1)));
        let [low, upper, mine]: [String; 3] = self.fresh_n();
        emit!(self, "selp.b32 {low}, {}, {}, {odd};", result[2], result[0]);
        emit!(self, "selp.b32 {upper}, {}, {}, {odd};", result[3], result[1]);
        emit!(self, "selp.b32 {mine}, {low}, {upper}, {high};");
        let theirs = self.exchange(&mine, 2);
        for name in &key {
            self.written(name);
        }
        emit!(self, "selp.b32 {}, {theirs}, {low}, {high};", key[0]);
        emit!(self, "selp.b32 {}, {upper}, {theirs}, {high};", key[1]);
        // (The registers of the program hold it again, and the Volta ones still do.)
        self.remember(Form::Half, &[key[0].as_str(), key[1].as_str()], &result);
    }

    /// One product in a chain: the single-instruction arrangement. One `mma.m8n8k4.row.col` per
    /// Turing `mma.m16n8k8`, the accumulator kept split across octets. At chain start the split
    /// accumulator is initialised (from zero or from the existing Turing values); at chain end it
    /// is combined and extracted to Turing layout.
    pub(super) fn mma_f32_chain(&mut self, d: &[&str], a: &[&str], b: &str, c: &[&str]) {
        let d_key: [String; 4] = std::array::from_fn(|i| d[i].to_string());
        let is_start = !self.active_chains.contains_key(&d_key);

        if is_start {
            // (A chain that is still open goes on from its split registers; one that was ended
            // left the accumulator in the registers of the program, and begins again from them.)
            let split: [String; 8] = self.fresh_n();
            if c.iter().all(|name| self.zeros.contains(*name)) {
                // From zero in every lane: zero in every layout.
                for register in &split {
                    emit!(self, "mov.b32 {register}, 0;");
                }
            } else {
                // Convert Turing C values to the split Volta layout. k_half=0 octets (q=0,1) get
                // the Turing values at the right positions; k_half=1 octets (q=2,3) get zero.
                //
                // For split register i_v at thread (k, h, q):
                //   j_src = 2*(q&1) + (i_v&1): which Turing register to read.
                //   M_src = variable_part + offset[i_v]:
                //     variable_part = 4*(k&1) + 16*h + (k>>1 & 1) (same for all i_v)
                //     offset = [0, 0, 8, 8, 2, 2, 10, 10] for i_v = 0..7.
                //
                // The source register depends on row_half: c[0] or c[2] for even i_v, c[1] or c[3]
                // for odd. The shuffle reads from another lane whose row_half may differ, so the
                // selp must be AFTER the shuffle, not before it. Each shuffle pair fetches c[0] and
                // c[2] (or c[1] and c[3]) from the source lane, then selp picks the right one.
                let is_kh1 = self.lane(Lane::Has(8));
                let is_rh1 = self.lane(Lane::Has(4));
                let lane_reg = self.lane(Lane::Id);

                // variable_part = 4*(lane & 1) + (lane & 16) + ((lane & 2) >> 1)
                let vp = self.fresh();
                let vt = self.fresh();
                emit!(self, "and.b32 {vp}, {lane_reg}, 1;");
                emit!(self, "shl.b32 {vp}, {vp}, 2;"); // 4*(lane & 1)
                emit!(self, "and.b32 {vt}, {lane_reg}, 16;");
                emit!(self, "or.b32 {vp}, {vp}, {vt};"); // + (lane & 16)
                emit!(self, "and.b32 {vt}, {lane_reg}, 2;");
                emit!(self, "shr.u32 {vt}, {vt}, 1;"); // ((lane & 2) >> 1)
                emit!(self, "or.b32 {vp}, {vp}, {vt};");

                // For each pair of registers (even, odd) sharing the same M_src:
                // offsets: (i_v=0,1 → 0), (i_v=2,3 → 8), (i_v=4,5 → 2), (i_v=6,7 → 10)
                let offsets = [0u32, 8, 2, 10];
                for (pair, &offset) in offsets.iter().enumerate() {
                    let even_iv = pair * 2;
                    let odd_iv = pair * 2 + 1;
                    let m_src = if offset == 0 {
                        vp.clone()
                    } else {
                        let r = self.fresh();
                        emit!(self, "add.u32 {r}, {vp}, {offset};");
                        r
                    };
                    // Shuffle BOTH Turing registers from the source lane, then selp by row_half.
                    // Even i_v: need c[0] or c[2]. Shuffle both, selp.
                    let from_rh0 = self.fresh();
                    let from_rh1 = self.fresh();
                    emit!(self, "shfl.sync.idx.b32 {from_rh0}, {}, {m_src}, 31, -1;", c[0]);
                    self.shuffles += 1;
                    emit!(self, "shfl.sync.idx.b32 {from_rh1}, {}, {m_src}, 31, -1;", c[2]);
                    self.shuffles += 1;
                    emit!(self, "selp.b32 {}, {from_rh1}, {from_rh0}, {is_rh1};", split[even_iv]);
                    // Odd i_v: need c[1] or c[3].
                    emit!(self, "shfl.sync.idx.b32 {from_rh0}, {}, {m_src}, 31, -1;", c[1]);
                    self.shuffles += 1;
                    emit!(self, "shfl.sync.idx.b32 {from_rh1}, {}, {m_src}, 31, -1;", c[3]);
                    self.shuffles += 1;
                    emit!(self, "selp.b32 {}, {from_rh1}, {from_rh0}, {is_rh1};", split[odd_iv]);
                    // Zero the k_half=1 octets.
                    emit!(self, "selp.b32 {}, 0, {}, {is_kh1};", split[even_iv], split[even_iv]);
                    emit!(self, "selp.b32 {}, 0, {}, {is_kh1};", split[odd_iv], split[odd_iv]);
                }
            }
            self.active_chains.insert(d_key.clone(), split);
        }

        let split = self.active_chains.get(&d_key).unwrap().clone();

        // A and B: in the layout of the single instruction when the plan had their `ldmatrix`
        // load them so (the registers are those the `ldmatrix` fills, wherever it is in the text:
        // it is the only statement that writes the registers of the program, so what it loaded
        // last is what a product reads); gathered from the registers of the program otherwise.
        let (aw0, aw1) = if self.plan.chain_a.contains(&(a[0], a[1])) {
            let loaded = self.operand(Form::ChainA, &[a[0], a[1]], false);
            (loaded[0].clone(), loaded[1].clone())
        } else {
            // Both registers from the lane that has the row, then the one of its half of the rows.
            let is_rh1 = self.lane(Lane::Has(4));
            let a0w0 = self.shuffle(a[0], Lane::ChainW0);
            let a1w0 = self.shuffle(a[1], Lane::ChainW0);
            let w0 = self.fresh();
            emit!(self, "selp.b32 {w0}, {a1w0}, {a0w0}, {is_rh1};");
            let a0w1 = self.shuffle(a[0], Lane::ChainW1);
            let a1w1 = self.shuffle(a[1], Lane::ChainW1);
            let w1 = self.fresh();
            emit!(self, "selp.b32 {w1}, {a1w1}, {a0w1}, {is_rh1};");
            (w0, w1)
        };
        let (bw0, bw1) = if self.plan.chain_b.contains(b) {
            let loaded = self.operand(Form::ChainB, &[b], false);
            (loaded[0].clone(), loaded[1].clone())
        } else {
            (self.shuffle(b, Lane::ChainW0), self.shuffle(b, Lane::ChainW1))
        };

        // One m8n8k4.row.col product on the split accumulator.
        let new_split: [String; 8] = self.fresh_n();
        emit!(
            self,
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \
             {{{}}}, {{{aw0}, {aw1}}}, {{{bw0}, {bw1}}}, {{{}}};",
            new_split.join(", "),
            split.join(", "),
        );
        self.active_chains.insert(d_key, new_split.clone());

        // The registers of the program are not written: they keep what they held before the chain
        // until `chain_end_one` gives them the accumulator. What was derived from them is no
        // longer the accumulator's.
        for name in d {
            self.written(name);
        }
    }

    /// End one open chain: give the accumulator back to the four registers of the program.
    ///
    /// Register `j` of the lane with bits `b4 b3 b2 b1 b0` is the sum of the two partial sums that
    /// the lanes `S` and `S ^ 8` hold in their split register `(j & 1) + 2 * b3 + 4 * b1`, where
    /// `S` has the bits `b4`, any, `j >> 1`, `b0`, `b2`. The lane read is the one of the two whose
    /// bit 3 is `b3`: it knows by its own bit 3 which register to give, and the lane 8 away which
    /// to add to it, so that one exchange and one shuffle bring what two lanes need; `b1` is
    /// chosen after the shuffle. 4 exchanges and 8 shuffles for the 4 registers.
    pub(super) fn chain_end_one(&mut self, d_key: [String; 4]) {
        let Some(split) = self.active_chains.remove(&d_key) else {
            return;
        };
        let (upper, second) = (self.lane(Lane::Has(8)), self.lane(Lane::Has(2)));
        // sums[2 * (j & 1) + b1], in the lane read.
        let mut sums = Vec::with_capacity(4);
        for parity in 0..2 {
            for pair in 0..2 {
                let (low, high) = (&split[parity + 4 * pair], &split[parity + 4 * pair + 2]);
                let [mine, given, sum]: [String; 3] = self.fresh_n();
                emit!(self, "selp.b32 {mine}, {high}, {low}, {upper};");
                // (To the lane 8 away, whose bit 3 is the other one.)
                emit!(self, "selp.b32 {given}, {low}, {high}, {upper};");
                let theirs = self.exchange(&given, 8);
                emit!(self, "add.f32 {sum}, {mine}, {theirs};");
                sums.push(sum);
            }
        }
        let from = [self.lane(Lane::ChainOut(false)), self.lane(Lane::ChainOut(true))];
        for name in &d_key {
            self.written(name);
        }
        for (j, name) in d_key.iter().enumerate() {
            let pair = [0, 1].map(|b1| {
                let fetched = self.fresh();
                emit!(
                    self,
                    "shfl.sync.idx.b32 {fetched}, {}, {}, 31, -1;",
                    sums[2 * (j & 1) + b1],
                    from[j >> 1]
                );
                self.shuffles += 1;
                fetched
            });
            emit!(self, "selp.b32 {name}, {}, {}, {second};", pair[1], pair[0]);
        }
        self.split();
    }

    /// End every open chain, and give back every f16 accumulator kept in Volta layout.
    pub(super) fn chain_end_all(&mut self) {
        let keys: Vec<[String; 4]> = self.active_chains.keys().cloned().collect();
        for key in keys {
            self.chain_end_one(key);
        }
        let keys: Vec<[String; 2]> = self.kept_halves.keys().cloned().collect();
        for key in keys {
            self.half_end_one(key);
        }
    }

    /// End the open chains, and give back the f16 accumulators kept in Volta layout, whose
    /// registers `operands` names (whole names: `%f1` is not named by `%f10`), but the one the
    /// statement goes on with when it is a product that takes its accumulator as it is kept
    /// (`product`: the accumulator of the registers of its first operand).
    ///
    /// A product goes on with an accumulator only if it reads it as its C and nowhere else:
    /// its A and its B are read from the registers of the program, which do not hold the
    /// accumulator while it is kept, and a C that has some of its registers and not all is read
    /// from them too. (Until 2026-11-19 the registers of the first operand were all that was
    /// looked at: `mma {d}, {d0, a1}, {b}, {d}` read in `%d0` what it held before the chain,
    /// with f32 chains and with f16 accumulators.)
    pub(super) fn chain_end_touching(&mut self, operands: &str, product: bool) {
        let named: HashSet<&str> = scan::names(operands).collect();
        let groups = scan::split_operands(operands);
        let own: Vec<&str> = if product {
            scan::names(first_operand(operands)).collect()
        } else {
            Vec::new()
        };
        // (D, A, B, C: what stands between the first and the last is read as it is.)
        let (between, last): (HashSet<&str>, Vec<&str>) = if product && groups.len() > 2 {
            (
                groups[1..groups.len() - 1]
                    .iter()
                    .flat_map(|group| scan::names(group))
                    .collect(),
                scan::names(groups[groups.len() - 1]).collect(),
            )
        } else {
            (HashSet::new(), Vec::new())
        };
        let its_own = |key: &[String]| {
            let names = || key.iter().map(String::as_str);
            product
                && names().eq(own.iter().copied())
                && !names().any(|name| between.contains(name))
                && (names().eq(last.iter().copied()) || !names().any(|name| last.contains(&name)))
        };
        let keys: Vec<[String; 4]> = self
            .active_chains
            .keys()
            .filter(|d| d.iter().any(|name| named.contains(name.as_str())) && !its_own(&d[..]))
            .cloned()
            .collect();
        let halves: Vec<[String; 2]> = self
            .kept_halves
            .keys()
            .filter(|d| d.iter().any(|name| named.contains(name.as_str())) && !its_own(&d[..]))
            .cloned()
            .collect();
        for key in keys {
            self.chain_end_one(key);
        }
        for key in halves {
            self.half_end_one(key);
        }
    }
}
