//! Instructions of later targets that have direct Volta equivalents: cache hints, bf16
//! conversions and arithmetic, redux, wide loads and stores, griddepcontrol, f32x2 operations,
//! and the other forms that `is_plain` recognises.

use std::fmt::Write as _;
use std::ops::Range;

use crate::scan;

use super::Unsupported;
use super::body::Body;
use super::parse::wide;
use super::types::{Fp8, all_ones, first_operand, place_of};

impl<'a> Body<'a> {
    /// Instructions of later targets with a plain equivalent (`is_plain`). The guard of the statement,
    /// if any, is kept on the statements that write what the program reads.
    pub(super) fn plain(
        &mut self,
        range: Range<usize>,
        opcode: &str,
        operands: &str,
        followed: bool,
    ) -> Result<(), Unsupported> {
        let unsupported = |what: &str| Unsupported(format!("{opcode}: {what}"));
        // The statement is `<guard><opcode><operands>;`.
        let at = opcode.as_ptr() as usize - self.text.as_ptr() as usize;
        let guard = self.text[range.start..at].to_string();
        let mut parts = scan::split_operands(operands);
        let mut words: Vec<&str> = opcode.split('.').collect();
        self.out.push_str(&self.text[self.copied..range.start]);
        self.copied = range.end;
        self.first_line = true;
        let memory = matches!(words[0], "ld" | "st");
        // What a load or a store says to the second-level cache changes nothing a program can see,
        // and this target's compiler refuses it: the size of a prefetch, the eviction priority, and
        // a cache policy, which is one more operand.
        let hinted = memory && words.contains(&"L2::cache_hint");
        if memory {
            words.retain(|word| {
                !matches!(
                    *word,
                    "L2::64B"
                        | "L2::128B"
                        | "L2::256B"
                        | "L2::cache_hint"
                        | "L2::evict_normal"
                        | "L2::evict_first"
                        | "L2::evict_last"
                )
            });
            if hinted && parts.pop().is_none() {
                return Err(unsupported("operands not recognised"));
            }
        }
        let operands = if hinted {
            format!(" {}", parts.join(", "))
        } else {
            operands.to_string()
        };
        let operands = operands.as_str();
        if words[0] == "mbarrier" || opcode.starts_with("cp.async.mbarrier.") {
            self.cta_barrier(&guard, &words, &parts)?;
        } else if words[0] == "griddepcontrol" {
            // Waits for, or releases, kernels launched as dependent on one another: without such
            // launches, which this GPU's driver does not have, there is nothing to wait for.
            self.nothing(followed);
        } else if words[0] == "createpolicy" {
            // A cache policy only ever reaches the hint of a load or a store, which is dropped.
            let Some(d) = parts.first().filter(|d| scan::register(d).is_some()) else {
                return Err(unsupported("operands not recognised"));
            };
            self.written(d);
            emit!(self, "{guard}mov.b64 {d}, 0;");
        } else if let [operation @ ("add" | "sub" | "mul" | "fma"), modifiers @ .., "f32x2"] = &words[..]
            && modifiers
                .iter()
                .all(|modifier| matches!(*modifier, "rn" | "rz" | "rm" | "rp" | "ftz"))
            && parts.len() == if *operation == "fma" { 4 } else { 3 }
            && parts.iter().all(|part| scan::register(part).is_some())
        {
            // Two f32 in a 64-bit register, each computed for itself with the same rounding.
            let single = format!("{}.f32", words[..words.len() - 1].join("."));
            let mut sources: Vec<[String; 2]> = Vec::new();
            for source in &parts[1..] {
                let pair: [String; 2] = self.fresh_n();
                emit!(self, "mov.b64 {{{}, {}}}, {source};", pair[0], pair[1]);
                sources.push(pair);
            }
            let results: [String; 2] = self.fresh_n();
            for (half, result) in results.iter().enumerate() {
                let operands: Vec<&str> = sources.iter().map(|pair| pair[half].as_str()).collect();
                emit!(self, "{single} {result}, {};", operands.join(", "));
            }
            let d = parts[0];
            self.written(d);
            emit!(self, "{guard}mov.b64 {d}, {{{}, {}}};", results[0], results[1]);
        } else if matches!(opcode, "min.f32" | "max.f32" | "min.ftz.f32" | "max.ftz.f32") {
            let [d, a, b, c] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let pair = self.fresh();
            emit!(self, "{opcode} {pair}, {a}, {b};");
            self.written(d);
            emit!(self, "{guard}{opcode} {d}, {pair}, {c};");
        } else if let ["min" | "max", "relu", "s32"] = words[..] {
            let [d, a, b] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let bound = self.fresh();
            emit!(self, "{}.s32 {bound}, {a}, {b};", words[0]);
            self.written(d);
            emit!(self, "{guard}max.s32 {d}, {bound}, 0;");
        } else if let ["cvt", rounding @ ("rn" | "rz"), "bf16", "f32"] = words[..] {
            let [d, a] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let bits = self.bf16(rounding, a);
            self.written(d);
            emit!(self, "{guard}cvt.u16.u32 {d}, {bits};");
        } else if let ["cvt", rounding @ ("rn" | "rz"), "bf16x2", "f32"] = words[..] {
            let [d, a, b] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let (upper, lower) = (self.bf16(rounding, a), self.bf16(rounding, b));
            emit!(self, "shl.b32 {upper}, {upper}, 16;");
            self.written(d);
            emit!(self, "{guard}or.b32 {d}, {upper}, {lower};");
        } else if let ["cvt", "f32", "bf16"] = words[..] {
            // Exact: a bf16 is the upper half of an f32.
            let [d, a] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let bits = self.fresh();
            emit!(self, "cvt.u32.u16 {bits}, {a};");
            emit!(self, "shl.b32 {bits}, {bits}, 16;");
            self.written(d);
            emit!(self, "{guard}mov.b32 {d}, {bits};");
        } else if let ["cvt", "rn", "bf16", "f16"] = words[..] {
            let [d, a] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let widened = self.fresh();
            emit!(self, "cvt.f32.f16 {widened}, {a};");
            let bits = self.bf16("rn", &widened);
            self.written(d);
            emit!(self, "{guard}cvt.u16.u32 {d}, {bits};");
        } else if let ["cvt", "rn", "f16", "bf16"] = words[..] {
            let [d, a] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let widened = self.bf16_to_f32(a);
            self.written(d);
            emit!(self, "{guard}cvt.rn.f16.f32 {d}, {widened};");
        } else if let [
            "cvt",
            "rn",
            destination @ ("f16x2" | "bf16x2"),
            kind @ ("e4m3x2" | "e5m2x2"),
        ] = words[..]
            && let Some(kind) = Fp8::named(kind)
        {
            // Exact: every FP8 value is a half. The upper byte gives the upper half.
            let [d, a] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let pair = self.fresh();
            emit!(self, "cvt.u32.u16 {pair}, {a};");
            let halves = self.fp8_pair(kind, &pair);
            if destination == "f16x2" {
                self.written(d);
                emit!(self, "{guard}mov.b32 {d}, {halves};");
            } else {
                // All finite FP8 values are exactly representable in both f16 and bf16.
                let (lo, hi) = (self.fresh_half(), self.fresh_half());
                let [low, high]: [String; 2] = self.fresh_n();
                emit!(self, "mov.b32 {{{lo}, {hi}}}, {halves};");
                emit!(self, "cvt.f32.f16 {low}, {lo};");
                emit!(self, "cvt.f32.f16 {high}, {hi};");
                let low = self.bf16("rz", &low);
                let high = self.bf16("rz", &high);
                emit!(self, "shl.b32 {high}, {high}, 16;");
                self.written(d);
                emit!(self, "{guard}or.b32 {d}, {high}, {low};");
            }
        } else if let [
            "cvt",
            "rn",
            "satfinite",
            kind @ ("e4m3x2" | "e5m2x2"),
            source @ ("f32" | "f16x2" | "bf16x2"),
        ] = words[..]
            && let Some(kind) = Fp8::named(kind)
        {
            // The first value (or the upper half) in the upper byte.
            let (upper, lower) = match (source, &parts[..]) {
                ("f32", [_, a, b]) => (a.to_string(), b.to_string()),
                ("bf16x2", [_, a]) => self.bf16x2_to_f32(a),
                ("f16x2", [_, a]) => {
                    // The halves as f32, exactly.
                    let halves = [self.fresh_half(), self.fresh_half()];
                    let [lower, upper]: [String; 2] = self.fresh_n();
                    emit!(self, "mov.b32 {{{}, {}}}, {a};", halves[0], halves[1]);
                    emit!(self, "cvt.f32.f16 {lower}, {};", halves[0]);
                    emit!(self, "cvt.f32.f16 {upper}, {};", halves[1]);
                    (upper, lower)
                }
                _ => return Err(unsupported("operands not recognised")),
            };
            let (upper, lower) = (self.fp8_byte(kind, &upper), self.fp8_byte(kind, &lower));
            emit!(self, "shl.b32 {upper}, {upper}, 8;");
            emit!(self, "or.b32 {upper}, {upper}, {lower};");
            let d = parts[0];
            self.written(d);
            emit!(self, "{guard}cvt.u16.u32 {d}, {upper};");
        } else if let ["cvt", "rn", "f16x2", "e2m1x2"] | ["cvt", "rn", "relu", "f16x2", "e2m1x2"] = words[..] {
            // Exact (`e2m1_pair`).
            let [d, a] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let halves = self.e2m1_pair(a, words[2] == "relu");
            self.written(d);
            emit!(self, "{guard}mov.b32 {d}, {halves};");
        } else if let ["cvt", "rn", "satfinite", "e2m1x2", "f32"]
        | ["cvt", "rn", "relu", "satfinite", "e2m1x2", "f32"] = words[..]
        {
            // The nearest E2M1 value of each of two floats, the first in the upper nibble of a
            // byte (`e2m1_nibble`).
            let [_, a, b] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let relu = words[2] == "relu";
            let upper = self.e2m1_nibble(a, relu);
            let lower = self.e2m1_nibble(b, relu);
            emit!(self, "shl.b32 {upper}, {upper}, 4;");
            emit!(self, "or.b32 {upper}, {upper}, {lower};");
            let d = parts[0];
            let narrow = if self.declarations.width(d).unwrap_or(16) <= 8 {
                "cvt.u8.u32"
            } else {
                "cvt.u16.u32"
            };
            self.written(d);
            emit!(self, "{guard}{narrow} {d}, {upper};");
        } else if let [
            "cvt",
            rounding @ ("rz" | "rp"),
            "satfinite",
            "ue8m0x2",
            source @ ("f32" | "bf16x2"),
        ] = words[..]
        {
            // The power of two at or below each of two values, or at or above it, as UE8M0 bytes,
            // the first in the upper byte (`ue8m0_byte`). (The compilers take no other rounding
            // for this type.)
            let (upper, lower) = match (source, &parts[..]) {
                ("f32", [_, a, b]) => (a.to_string(), b.to_string()),
                ("bf16x2", [_, a]) => {
                    let (hi, lo) = self.bf16x2_to_f32(a);
                    (hi, lo)
                }
                _ => return Err(unsupported("operands not recognised")),
            };
            let upper = self.ue8m0_byte(rounding == "rp", &upper);
            let lower = self.ue8m0_byte(rounding == "rp", &lower);
            emit!(self, "shl.b32 {upper}, {upper}, 8;");
            emit!(self, "or.b32 {upper}, {upper}, {lower};");
            let d = parts[0];
            self.written(d);
            emit!(self, "{guard}cvt.u16.u32 {d}, {upper};");
        } else if let ["cvt", "rn", "bf16x2", "ue8m0x2"] = words[..] {
            // Exact (`ue8m0_pair`).
            let [d, a] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let halves = self.ue8m0_pair(a);
            self.written(d);
            emit!(self, "{guard}mov.b32 {d}, {halves};");
        } else if let ["cvt", "rn", "tf32", "f32"] = words[..] {
            // An f32 with 10 bits of mantissa: rounded on its 13 low bits, to nearest and ties to
            // even (add 0xfff and the lowest bit that stays), which carries into the exponent as it
            // must. A NaN stays one whatever its low bits.
            let [d, a] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let [bits, odd]: [String; 2] = self.fresh_n();
            self.scratch_predicate = true;
            emit!(self, "mov.b32 {bits}, {a};");
            emit!(self, "shr.u32 {odd}, {bits}, 13;");
            emit!(self, "and.b32 {odd}, {odd}, 1;");
            emit!(self, "add.u32 {odd}, {odd}, 4095;");
            emit!(self, "add.u32 {bits}, {bits}, {odd};");
            emit!(self, "and.b32 {bits}, {bits}, 0xffffe000;");
            emit!(self, "setp.nan.f32 %vf_n, {a}, {a};");
            emit!(self, "selp.b32 {bits}, 0x7fffe000, {bits}, %vf_n;");
            self.written(d);
            emit!(self, "{guard}mov.b32 {d}, {bits};");
        } else if let ["neg", "bf16"] = words[..] {
            // Flip the sign bit of a bf16 (bit 15 of a 16-bit register).
            let [d, a] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let bits = self.fresh();
            emit!(self, "cvt.u32.u16 {bits}, {a};");
            emit!(self, "xor.b32 {bits}, {bits}, 0x8000;");
            self.written(d);
            emit!(self, "{guard}cvt.u16.u32 {d}, {bits};");
        } else if let [op @ ("max" | "min"), "NaN", "bf16"] = words[..] {
            let [d, a, b] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let a = self.bf16_to_f32(a);
            let b = self.bf16_to_f32(b);
            let result = self.fresh();
            self.bound(&format!("{op}.f32"), &result, &a, &b);
            emit!(self, "shr.u32 {result}, {result}, 16;");
            self.written(d);
            emit!(self, "{guard}cvt.u16.u32 {d}, {result};");
        } else if let [op @ ("max" | "min"), "bf16x2"] | [op @ ("max" | "min"), "NaN", "bf16x2"] = words[..] {
            let [d, a, b] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let (a_hi, a_lo) = self.bf16x2_to_f32(a);
            let (b_hi, b_lo) = self.bf16x2_to_f32(b);
            let [high, low]: [String; 2] = self.fresh_n();
            let plain = format!("{op}.f32");
            if words.len() == 3 {
                self.bound(&plain, &high, &a_hi, &b_hi);
                self.bound(&plain, &low, &a_lo, &b_lo);
            } else {
                emit!(self, "{plain} {high}, {a_hi}, {b_hi};");
                emit!(self, "{plain} {low}, {a_lo}, {b_lo};");
            }
            // Each result is an input bf16 or a canonical NaN. Pack the two high halfwords.
            self.written(d);
            emit!(self, "{guard}prmt.b32 {d}, {low}, {high}, 0x7632;");
        } else if let ["abs", kind @ ("bf16" | "bf16x2")] = words[..] {
            let [d, a] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            self.written(d);
            if kind == "bf16" {
                emit!(self, "{guard}and.b16 {d}, {a}, 0x7fff;");
            } else {
                emit!(self, "{guard}and.b32 {d}, {a}, 0x7fff7fff;");
            }
        } else if let [op @ ("add" | "sub"), "rn", "f32", "bf16"] = words[..] {
            // Only the first input is bf16. The accumulator and destination are f32.
            let [d, a, b] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let widened = self.bf16_to_f32(a);
            self.written(d);
            emit!(self, "{guard}{op}.rn.f32 {d}, {widened}, {b};");
        } else if let [op @ ("add" | "sub" | "mul"), "bf16"] | [op @ ("add" | "sub" | "mul"), "rn", "bf16"] = words[..]
        {
            // The wider precision holds all significant product bits, and all sum bits that can
            // change a bf16 rounding decision.
            let [d, a, b] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let a_f32 = self.bf16_to_f32(a);
            let b_f32 = self.bf16_to_f32(b);
            let result = self.fresh();
            emit!(self, "{op}.rn.f32 {result}, {a_f32}, {b_f32};");
            let bits = self.bf16("rn", &result);
            self.written(d);
            emit!(self, "{guard}cvt.u16.u32 {d}, {bits};");
        } else if let [op @ ("add" | "sub" | "mul"), "bf16x2"] | [op @ ("add" | "sub" | "mul"), "rn", "bf16x2"] =
            words[..]
        {
            let [d, a, b] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let (a_hi, a_lo) = self.bf16x2_to_f32(a);
            let (b_hi, b_lo) = self.bf16x2_to_f32(b);
            let [high, low]: [String; 2] = self.fresh_n();
            emit!(self, "{op}.rn.f32 {high}, {a_hi}, {b_hi};");
            emit!(self, "{op}.rn.f32 {low}, {a_lo}, {b_lo};");
            let hi_bits = self.bf16("rn", &high);
            let lo_bits = self.bf16("rn", &low);
            emit!(self, "shl.b32 {hi_bits}, {hi_bits}, 16;");
            self.written(d);
            emit!(self, "{guard}or.b32 {d}, {hi_bits}, {lo_bits};");
        } else if let ["fma", "rn", "bf16"] = words[..] {
            // a * b can be a halfway point of bf16 and c anything, however small: the f32 result is
            // rounded to odd before the rounding to bf16 (`odd`).
            let [d, a, b, c] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let a_f32 = self.bf16_to_f32(a);
            let b_f32 = self.bf16_to_f32(b);
            let c_f32 = self.bf16_to_f32(c);
            let [down, up]: [String; 2] = self.fresh_n();
            emit!(self, "fma.rm.f32 {down}, {a_f32}, {b_f32}, {c_f32};");
            emit!(self, "fma.rp.f32 {up}, {a_f32}, {b_f32}, {c_f32};");
            let result = self.odd(&down, &up);
            let bits = self.bf16("rn", &result);
            self.written(d);
            emit!(self, "{guard}cvt.u16.u32 {d}, {bits};");
        } else if let ["fma", "rn", "bf16x2"] = words[..] {
            let [d, a, b, c] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let (a_hi, a_lo) = self.bf16x2_to_f32(a);
            let (b_hi, b_lo) = self.bf16x2_to_f32(b);
            let (c_hi, c_lo) = self.bf16x2_to_f32(c);
            let [down_hi, up_hi, down_lo, up_lo]: [String; 4] = self.fresh_n();
            emit!(self, "fma.rm.f32 {down_hi}, {a_hi}, {b_hi}, {c_hi};");
            emit!(self, "fma.rp.f32 {up_hi}, {a_hi}, {b_hi}, {c_hi};");
            emit!(self, "fma.rm.f32 {down_lo}, {a_lo}, {b_lo}, {c_lo};");
            emit!(self, "fma.rp.f32 {up_lo}, {a_lo}, {b_lo}, {c_lo};");
            let res_hi = self.odd(&down_hi, &up_hi);
            let res_lo = self.odd(&down_lo, &up_lo);
            let hi_bits = self.bf16("rn", &res_hi);
            let lo_bits = self.bf16("rn", &res_lo);
            emit!(self, "shl.b32 {hi_bits}, {hi_bits}, 16;");
            self.written(d);
            emit!(self, "{guard}or.b32 {d}, {hi_bits}, {lo_bits};");
        } else if let ["cvt", "rn", "bf16", "f64"] = words[..] {
            // An f64 can be next to a halfway point of bf16, nearer than an f32 can tell: to f32
            // with a rounding to odd (`odd`), then to bf16.
            let [d, a] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let [down, up]: [String; 2] = self.fresh_n();
            emit!(self, "cvt.rm.f32.f64 {down}, {a};");
            emit!(self, "cvt.rp.f32.f64 {up}, {a};");
            let f32_val = self.odd(&down, &up);
            let bits = self.bf16("rn", &f32_val);
            self.written(d);
            emit!(self, "{guard}cvt.u16.u32 {d}, {bits};");
        } else if let [
            "setp",
            cmp @ ("eq" | "ne" | "lt" | "le" | "gt" | "ge" | "equ" | "neu" | "ltu" | "leu" | "gtu" | "geu" | "num"
            | "nan"),
            ref boolean @ ..,
            "bf16",
        ] = words[..]
        {
            // Bit-exact widening keeps subnormals, signed zeros and NaNs for ordered and
            // unordered comparisons. PTX uses the same Boolean operators on f32 predicates.
            let (d, a, b, suffix, extra) = match (boolean, parts.as_slice()) {
                ([], &[d, a, b]) => (d, a, b, String::new(), String::new()),
                ([op @ ("and" | "or" | "xor")], &[d, a, b, c]) => (d, a, b, format!(".{op}"), format!(", {c}")),
                _ => return Err(unsupported("operands or Boolean operation not recognised")),
            };
            let a_f32 = self.bf16_to_f32(a);
            let b_f32 = self.bf16_to_f32(b);
            self.written(d);
            emit!(self, "{guard}setp.{cmp}{suffix}.f32 {d}, {a_f32}, {b_f32}{extra};");
        } else if let [
            "set",
            cmp @ ("eq" | "ne" | "lt" | "le" | "gt" | "ge"),
            "bf16x2",
            "bf16x2",
        ] = words[..]
        {
            // Compare two pairs of bf16 values. For each element: bf16 1.0 (0x3f80) if the
            // comparison is true, bf16 0.0 (0x0000) if false. Result is a 32-bit register with two
            // bf16 elements.
            let [d, a, b] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let (a_hi, a_lo) = self.bf16x2_to_f32(a);
            let (b_hi, b_lo) = self.bf16x2_to_f32(b);
            self.scratch_predicate = true;
            let [hi_result, lo_result]: [String; 2] = self.fresh_n();
            emit!(self, "setp.{cmp}.f32 %vf_n, {a_hi}, {b_hi};");
            emit!(self, "selp.u32 {hi_result}, 0x3f800000, 0, %vf_n;");
            emit!(self, "setp.{cmp}.f32 %vf_n, {a_lo}, {b_lo};");
            emit!(self, "selp.u32 {lo_result}, 0x3f800000, 0, %vf_n;");
            // Pack: upper bf16 is bits 31..16 of hi_result (0x3f80 or 0x0000), lower is from lo_result.
            emit!(self, "and.b32 {hi_result}, {hi_result}, 0xffff0000;");
            emit!(self, "shr.u32 {lo_result}, {lo_result}, 16;");
            self.written(d);
            emit!(self, "{guard}or.b32 {d}, {hi_result}, {lo_result};");
        } else if words[0] == "atom"
            && words.contains(&"add")
            && words.contains(&"noftz")
            && matches!(words.last(), Some(&"bf16") | Some(&"bf16x2"))
        {
            // atom{.sem}{.scope}{.space}.add.noftz.bf16/bf16x2 d, [a], b: add atomically through
            // a CAS loop. Every word of the opcode must be accounted for.
            let [d, addr, b] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let pair = words.last() == Some(&"bf16x2");
            let mut sem = "";
            let mut scope = "";
            let mut space = "";
            for &word in &words[1..] {
                match word {
                    "relaxed" | "acquire" | "release" | "acq_rel" => sem = word,
                    "cta" | "gpu" | "sys" => scope = word,
                    // .cluster does not exist on sm_70: widen to .gpu.
                    "cluster" => scope = "gpu",
                    "global" => space = ".global",
                    "shared" => space = ".shared",
                    w if w.starts_with("shared::") => space = ".shared",
                    "add" | "noftz" | "bf16" | "bf16x2" => {}
                    w if w.contains("L2::") => {} // cache hint, dropped
                    _ => return Err(unsupported(&format!("atom.add.noftz.bf16: unknown qualifier .{word}"))),
                }
            }
            let sem_str = if sem.is_empty() {
                String::new()
            } else {
                format!(".{sem}")
            };
            let scope_str = if scope.is_empty() {
                String::new()
            } else {
                format!(".{scope}")
            };
            self.atomic_add_bf16(&guard, pair, space, &sem_str, &scope_str, [d, addr, b], followed);
        } else if let [
            "redux",
            "sync",
            operation @ ("add" | "min" | "max" | "and" | "or" | "xor"),
            ref qualifiers @ ..,
            kind,
        ] = words[..]
            && (qualifiers.is_empty() || (qualifiers == ["NaN"] && kind == "f32" && matches!(operation, "min" | "max")))
        {
            // The member mask names the whole warp. Some of its threads may have exited.
            let [d, a, mask] = parts[..] else {
                return Err(unsupported("operands not recognised"));
            };
            let whole_warp = all_ones(mask) || self.all_ones.contains(mask) || self.plan.ones.contains(mask);
            if !guard.is_empty() || !whole_warp {
                return Err(unsupported("no Volta sequence for this form"));
            }
            let instruction = match operation {
                "and" | "or" | "xor" => format!("{operation}.b32"),
                _ => format!("{operation}.{kind}"),
            };
            let value = self.fresh();
            let [live, remaining, peer, bit, other]: [String; 5] = self.fresh_n();
            let lane = self.lane(super::lane::Lane::Id);
            let n = self.current_index;
            self.scratch_predicate = true;
            emit!(self, "mov.b32 {value}, {a};");
            emit!(self, "vote.sync.ballot.b32 {live}, 1, -1;");
            emit!(self, "setp.ne.u32 %vf_n, {live}, -1;");
            emit!(self, "@%vf_n bra.uni $vf_partial{n};");
            for distance in [16, 8, 4, 2, 1] {
                let other = self.exchange(&value, distance);
                emit!(self, "{instruction} {value}, {value}, {other};");
            }
            emit!(self, "bra.uni $vf_reduced{n};");

            // Start with this lane's own input, then visit each surviving peer once.
            // Every lane iterates popc(live) - 1 times and shuffles with the same mask.
            let _ = write!(self.out, "\n$vf_partial{n}:");
            emit!(self, "shl.b32 {bit}, 1, {lane};");
            emit!(self, "xor.b32 {remaining}, {live}, {bit};");
            let _ = write!(self.out, "\n$vf_peers{n}:");
            emit!(self, "setp.eq.u32 %vf_n, {remaining}, 0;");
            emit!(self, "@%vf_n bra.uni $vf_reduced{n};");
            emit!(self, "bfind.u32 {peer}, {remaining};");
            emit!(self, "shfl.sync.idx.b32 {other}, {a}, {peer}, 31, {live};");
            emit!(self, "{instruction} {value}, {value}, {other};");
            emit!(self, "shl.b32 {bit}, 1, {peer};");
            emit!(self, "xor.b32 {remaining}, {remaining}, {bit};");
            emit!(self, "bra.uni $vf_peers{n};");
            let _ = write!(self.out, "\n$vf_reduced{n}:");
            self.shuffles = 0;
            self.written(d);
            if qualifiers.is_empty() {
                emit!(self, "mov.b32 {d}, {value};");
            } else {
                // The reduction above ignores NaNs. The .NaN form propagates a NaN from any live lane.
                emit!(self, "setp.nan.f32 %vf_n, {a}, {a};");
                emit!(self, "vote.sync.any.pred %vf_n, %vf_n, -1;");
                emit!(self, "selp.f32 {d}, 0f7FFFFFFF, {value}, %vf_n;");
            }
        } else if memory && wide(&words).is_some() {
            // More than 128 bits at once: two halves, each aligned as the whole was.
            let (count, bytes) = wide(&words).unwrap_or_default();
            let (registers, place) = if words[0] == "ld" {
                (parts.first(), parts.get(1))
            } else {
                (parts.get(1), parts.first())
            };
            let (Some(registers), Some((base, offset))) = (
                registers
                    .and_then(|list| scan::vector(list))
                    .filter(|list| list.len() == count),
                place.and_then(|place| place_of(place)),
            ) else {
                return Err(unsupported("operands not recognised"));
            };
            let half = words[..words.len() - 2].join(".");
            let kind = words[words.len() - 1];
            if words[0] == "ld" {
                // A vector load reads all sources then writes all destinations, but the split
                // into two instructions breaks that: the first half could overwrite the base
                // address register before the second half reads it. When the base is a
                // destination of the first half only, emit the second half first (the order of
                // elements within a vector load is unspecified, PTX ISA section 8.2.4). When it
                // is a destination of both halves, there is no safe order.
                let first_half = &registers[..count / 2];
                let second_half = &registers[count / 2..];
                let base_in_first = first_half.contains(&base);
                let base_in_second = second_half.contains(&base);
                if base_in_first && base_in_second {
                    return Err(unsupported(
                        "base address is a destination of both halves of a wide load",
                    ));
                }
                let order: [usize; 2] = if base_in_first { [1, 0] } else { [0, 1] };
                for &half_idx in &order {
                    let part = &registers[half_idx * count / 2..(half_idx + 1) * count / 2];
                    let address = format!("[{base}+{}]", offset + (half_idx * count / 2) as u32 * bytes);
                    let list = format!("{{{}}}", part.join(", "));
                    for name in part {
                        self.written(name);
                    }
                    emit!(self, "{guard}{half}.v{}.{kind} {list}, {address};", count / 2);
                }
            } else {
                for (index, part) in registers.chunks(count / 2).enumerate() {
                    let address = format!("[{base}+{}]", offset + (index * count / 2) as u32 * bytes);
                    let list = format!("{{{}}}", part.join(", "));
                    emit!(self, "{guard}{half}.v{}.{kind} {address}, {list};", count / 2);
                }
            }
        } else if memory {
            // Without the hints (above), and without `.volatile` on memory of the thread itself,
            // which this target's compiler refuses too.
            let local = words.contains(&"local");
            let kept: Vec<&str> = words
                .iter()
                .copied()
                .filter(|word| !(local && *word == "volatile"))
                .collect();
            if kept
                .iter()
                .any(|word| word.contains("::") && !word.starts_with("shared::"))
            {
                return Err(unsupported("no Volta sequence for this form"));
            }
            if words[0] == "ld" {
                for name in scan::names(first_operand(operands)) {
                    self.written(name);
                }
            }
            emit!(self, "{guard}{}{operands};", kept.join("."));
        } else {
            return Err(unsupported("no Volta sequence for this form"));
        }
        self.counts.other += 1;
        Ok(())
    }
}
