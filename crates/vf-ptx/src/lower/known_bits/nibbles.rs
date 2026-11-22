//! Prove that both halves of an MMA word are exact BF16 integers in 0..=15.
use std::collections::{HashMap, HashSet};

use super::Word;
use crate::lower::types::{Declarations, first_operand, type_width};
use crate::scan::{self, Item, Kind};

fn small_integer(value: f32) -> bool {
    value.is_sign_positive() && value <= 15.0 && value.fract() == 0.0
}

fn half(word: u32, shift: u32) -> u16 {
    (word >> shift) as u16
}

fn value(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// The product and the cancellation are exact in BF16. Enumerating at most sixteen
/// encodings per lane proves integrality as well as range; endpoints alone cannot.
fn packed_sum(a: Word, b: Word, c: Word) -> bool {
    let words = [a, b, c];
    let Some((products, index)) = Word::exact_sum(words) else {
        return false;
    };
    [0, 16].into_iter().zip(products).all(|(shift, product)| {
        let lo = half(words[index].low, shift);
        let hi = half(words[index].high, shift);
        hi - lo <= 15 && (lo..=hi).all(|bits| small_integer(f32::from_bits(product) + value(bits)))
    })
}

pub(in crate::lower) fn at_products(text: &str, items: &[Item], mma: &str) -> HashSet<usize> {
    let mut shadow = Declarations::default();
    let mut other_names = HashSet::new();
    let mut depth = 0;
    for item in items {
        match &item.kind {
            // The scanner consumes the rest of a .loc/.file line. An unrecognised line
            // can contain another statement, even a declaration that shadows a register.
            Kind::Line if !crate::deps::debug_location(&text[item.range.clone()]) => return HashSet::new(),
            Kind::Open => depth += 1,
            Kind::Close => depth -= 1,
            Kind::Statement { opcode, operands, .. } if depth > 0 => {
                let op = &text[opcode.clone()];
                let args = &text[operands.clone()];
                if op == ".reg" {
                    // Vector or otherwise unfamiliar declarations may hide another register.
                    if args.split_whitespace().next().and_then(type_width).is_none() {
                        return HashSet::new();
                    }
                    shadow.declare(args);
                } else if op.starts_with('.') {
                    other_names.extend(scan::names(args));
                }
            }
            _ => {}
        }
    }
    let mut words: HashMap<&str, Word> = HashMap::new();
    let mut small: HashSet<&str> = HashSet::new();
    let mut proved = HashSet::new();
    let unshadowed = |name: &str| shadow.width(name).is_none() && !other_names.contains(name);
    let mut constants = super::literals::immutable(text, items);
    constants.retain(|name, _| unshadowed(name));
    for (index, item) in items.iter().enumerate() {
        let (guarded, op, args) = match &item.kind {
            Kind::Statement {
                guarded,
                opcode,
                operands,
            } => (*guarded, &text[opcode.clone()], &text[operands.clone()]),
            Kind::Label => {
                words.clear();
                small.clear();
                continue;
            }
            Kind::Open | Kind::Close => {
                words.retain(|name, _| unshadowed(name));
                small.retain(|name| unshadowed(name));
                continue;
            }
            Kind::Line => continue,
        };
        if op == ".reg" {
            continue;
        }
        // Only these instructions have known destinations here. Calls, branches,
        // unknown directives and instructions with implicit outputs end all facts.
        if op != mma
            && !matches!(
                op,
                "mov.b32" | "mov.s32" | "mov.u32" | "mov.f32" | "and.b32" | "or.b32" | "shr.u32" | "fma.rn.bf16x2"
            )
        {
            words.clear();
            small.clear();
            continue;
        }
        let args_list = scan::split_operands(args);
        let word = |s: &str| Word::get(&words, &constants, s);
        let known = |s: &str| {
            let w = word(s);
            small.contains(s) || (w.low == w.high && [0, 16].into_iter().all(|i| small_integer(value(half(w.low, i)))))
        };
        if !guarded
            && op == mma
            && let [_, _, b, _] = args_list[..]
            && let Some(regs) = scan::vector(b)
            && regs.len() == 2
            && regs.iter().all(|r| known(r))
        {
            proved.insert(index);
        }
        // Read all operands before killing the destinations, including an aliased input.
        let is_small = !guarded
            && match (op, &args_list[..]) {
                ("mov.b32" | "mov.u32" | "mov.s32", [_, x]) => known(x),
                ("fma.rn.bf16x2", [_, a, b, c]) => packed_sum(word(a), word(b), word(c)),
                _ => false,
            };
        let bits = (!guarded)
            .then(|| Word::definition(&words, &constants, op, args))
            .flatten();
        for dest in scan::names(first_operand(args)) {
            words.remove(dest);
            small.remove(dest);
        }
        if let Some((dest, w)) = bits {
            words.insert(dest, w);
        }
        if is_small && let Some(dest) = args_list.first().and_then(|d| scan::register(d)) {
            small.insert(dest);
        }
    }
    proved
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCT: &str = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32";
    const DECODE: &str = "
        and.b32 %n,%raw,0x000f000f;
        or.b32 %e,%n,0x43004300;
        mov.u32 %k,0x43004300;
        {.reg .b32 tmp; mov.b32 tmp,0xbf80bf80U; fma.rn.bf16x2 %w,%k,tmp,%e;}
        mov.b32 %other,%w;
    ";

    fn proved(decode: &str, gap: &str) -> bool {
        let s = format!(
            "{{.reg .pred %p; .reg .b32 a,b,c,d,%n,%raw,%e,%k,%w,%other; {decode} {gap} {PRODUCT} {{a,b,c,d}},{{a,b,c,d}},{{%w,%other}},{{a,b,c,d}};}}"
        );
        let items = scan::items(s.as_bytes(), 0..s.len()).unwrap();
        at_products(&s, &items, PRODUCT).len() == 1
    }

    #[test]
    fn proof_is_local_to_an_uninterrupted_set_of_reaching_definitions() {
        assert!(proved(DECODE, ""));
        assert!(proved(DECODE, ".loc 1 2 0\n"));
        assert!(proved(DECODE, "{.reg .b32 tmp; mov.b32 tmp,0xffffffff;}"));
        assert!(proved(
            &DECODE
                .replace("%n,%raw", "%n,%n")
                .replace("%w,%k,tmp,%e", "%e,%k,tmp,%e")
                .replace("%other,%w", "%w,%e; mov.b32 %other,%w"),
            ""
        ));
        for gap in [
            "{.reg .v2 .b32 %w;}",
            "join:",
            ".loc 1 2 0 mov.b32 %w,%raw;\n",
            ".file 1 \"x.cu\" @%p mov.b32 %w,%raw;\n",
            "bra next;",
            "call f,();",
            "opaque.op %unused,%w;",
            "@%p mov.u32 %w,0;",
            "mov.b32 {%w,%other},%rd;",
            "{.reg .b32 %w; mov.b32 %other,%w;}",
            "{.reg .b32 %w<2>; mov.b32 %other,%w1;}",
            "fma.rn.bf16x2 %w,%k,%k,%e;",
            "mov.b32 %w,%raw;",
        ] {
            assert!(!proved(DECODE, gap), "{gap}");
        }
        for bad in [
            DECODE.replace("0x000f000f", "0x001f000f"),
            DECODE.replace("0x43004300", "0x42804280"),
            DECODE.replace("0xbf80bf80U", "0xbf00bf80U"),
            DECODE.replace("fma.rn.bf16x2 %w", "@%p fma.rn.bf16x2 %w"),
            DECODE.replace("or.b32 %e", "$join: or.b32 %e"),
            DECODE.replace(
                "fma.rn.bf16x2 %w,%k,tmp,%e;",
                "{.reg .b32 tmp; mov.u32 tmp,1; fma.rn.bf16x2 %w,%k,tmp,%e;}",
            ),
            DECODE.replace("%k,tmp,%e", "%k,tmp,%raw"),
            DECODE.replace("mov.u32 %k", "@%p mov.u32 %k"),
        ] {
            assert!(!proved(&bad, ""), "{bad}");
        }
    }

    #[test]
    fn every_accepted_halfword_reconstructs_a_nonnegative_small_integer() {
        let word = |bits| Word { low: bits, high: bits };
        let one = word(0x3f803f80);
        let negative_offset = word(0xc300c300);
        let product = word(0x43004300);
        let minus = word(0xbf80bf80);
        // Check the full domain of either half while the other half is known-good.
        for h in 0..=u16::MAX {
            for (lo, hi) in [(h, 0x4307), (0x4307, h)] {
                let c = word(u32::from(lo) | u32::from(hi) << 16);
                let exact = value(h) - 128.0;
                let expected = (0..=15).any(|n| exact.to_bits() == (n as f32).to_bits());
                assert_eq!(packed_sum(product, minus, c), expected, "{lo:04x}, {hi:04x}");
                assert_eq!(packed_sum(c, one, negative_offset), expected);
                assert_eq!(packed_sum(one, c, negative_offset), expected);
            }
        }
        for boost in [0u16, 1, 7, 15] {
            assert!(packed_sum(
                product,
                minus,
                Word {
                    low: 0x43004300,
                    high: 0x43004300 | u32::from(boost) << 16 | 15
                }
            ));
        }
        assert!(!packed_sum(
            product,
            minus,
            Word {
                low: 0x43004300,
                high: 0x43004310
            }
        ));
    }
}
