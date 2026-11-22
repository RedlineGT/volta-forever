//! Parsing of tensor-core instructions and instruction classification: which opcodes this
//! module rewrites, which it refuses, and how their operands are taken apart.

use std::collections::HashMap;

use crate::scan;

use super::types::{Element, Fp8, ScaledKind, address, integer, place_of, type_width};
use super::{LATER_TYPES, Unsupported};

/// A tensor-core instruction this module rewrites, with its operands.
pub(super) enum Tensor<'a> {
    /// FP16 `mma.m16n8k8` or `mma.m16n8k16`, with f16 accumulators if `half`, f32 otherwise. The
    /// second is two of the first in a row: its A registers 0-1 and its B register 0 are the operands
    /// of an `m16n8k8` product for k 0..7, its A registers 2-3 and B register 1 those for k 8..15.
    Product {
        half: bool,
        d: Vec<&'a str>,
        /// Two registers per `m16n8k8` product.
        a: Vec<&'a str>,
        /// One register per `m16n8k8` product.
        b: Vec<&'a str>,
        c: Vec<&'a str>,
    },
    /// Integer `mma.m8n8k16`; `types` are the `dp4a` operand types for A and B.
    Bytes {
        types: String,
        satfinite: bool,
        d: Vec<&'a str>,
        a: &'a str,
        b: &'a str,
        c: Vec<&'a str>,
    },
    /// Integer `mma.m16n8k16` or `mma.m16n8k32` on int8: decomposes into Turing `mma.m8n8k16`
    /// products. Two A registers and one B for m16n8k16, four A and two B for m16n8k32.
    AmpereBytes {
        types: String,
        satfinite: bool,
        d: Vec<&'a str>,
        a: Vec<&'a str>,
        b: Vec<&'a str>,
        c: Vec<&'a str>,
    },
    /// FP8 `mma.m16n8k32` with f32 accumulators: four A registers and two B registers of four bytes.
    Fp8Product {
        a_type: Fp8,
        b_type: Fp8,
        d: Vec<&'a str>,
        a: Vec<&'a str>,
        b: Vec<&'a str>,
        c: Vec<&'a str>,
    },
    /// A product whose sums each thread computes itself, accumulators of the type of the sums:
    /// `mma.m16n8k16` and `m16n8k8` on bf16, `m16n8k8` and `m16n8k4` on tf32 (four f32 accumulators),
    /// `m8n8k4` on f64 (two). Each register of `b` holds a block of k, and `a` holds, for each block,
    /// one register per half of the rows.
    Sums {
        element: Element,
        d: Vec<&'a str>,
        a: Vec<&'a str>,
        b: Vec<&'a str>,
        c: Vec<&'a str>,
    },
    /// Block-scaled MMA: `kind::mxf4`, `kind::mxf4nvf4` (m16n8k64, E2M1) or `kind::mxf8f6f4`
    /// (m16n8k32, E4M3/E5M2), with UE8M0 or UE4M3 scales. D = (A * scale_A) * (B * scale_B) + C,
    /// the K dimension split into blocks, each block of A/B multiplied by its scale.
    BlockScaled {
        kind: ScaledKind,
        d: Vec<&'a str>,
        a: Vec<&'a str>,
        b: Vec<&'a str>,
        c: Vec<&'a str>,
        scale_a: &'a str,
        byte_id_a: u32,
        thread_id_a: u32,
        scale_b: &'a str,
        byte_id_b: u32,
        thread_id_b: u32,
    },
    /// `ldmatrix` of `r.len()` matrices, from the address `[base]` or `[base+offset]`.
    Load {
        transposed: bool,
        shared: bool,
        r: Vec<&'a str>,
        base: &'a str,
        offset: Option<&'a str>,
    },
    /// One 32-bit fragment per 8x8 matrix stored in CTA shared memory.
    Store {
        transposed: bool,
        src: Vec<&'a str>,
        base: &'a str,
        offset: Option<&'a str>,
    },
    /// `movmatrix`.
    Transpose { d: &'a str, a: &'a str },
    /// `cp.async` of `size` bytes (4, 8 or 16) from global to shared memory, addresses as
    /// (register, offset). `amount`: the optional operand that says how much of the source is copied
    /// (the rest is zeros), or, if it is a predicate, that the source is ignored.
    Copy {
        size: u32,
        to: (&'a str, u32),
        from: (&'a str, u32),
        amount: Option<&'a str>,
    },
    /// `cp.async.commit_group`, `wait_group`, `wait_all`: nothing is left to wait for.
    Copied,
    /// `cvt.rn.f16x2.f32 d, a, b` (or `.rz`): `a` converted in the upper half of `d`, `b` in the lower.
    Pack {
        rounding: &'static str,
        d: &'a str,
        a: &'a str,
        b: &'a str,
    },
    /// `min.NaN.f32 d, a, b` or `max.NaN.f32`: `plain` is the instruction without `.NaN`.
    Bound {
        plain: &'static str,
        d: &'a str,
        a: &'a str,
        b: &'a str,
        third: Option<&'a str>,
    },
}

/// The opcode is a tensor-core instruction that Volta lacks.
pub(super) fn is_tensor(opcode: &str) -> bool {
    let volta = opcode.starts_with("mma.sync.aligned.m8n8k4.") && !opcode.contains(".f64");
    !volta
        && ["mma.", "ldmatrix.", "movmatrix.", "stmatrix."]
            .iter()
            .any(|prefix| opcode.starts_with(prefix))
}

/// The opcode is another instruction this module rewrites: an asynchronous copy, a conversion to
/// packed floats, a bound that returns NaN.
pub(super) fn is_later(opcode: &str) -> bool {
    opcode.starts_with("cp.async.")
        || (opcode.starts_with("cvt.") && opcode.contains("x2.f32"))
        || opcode.contains(".NaN.")
}

/// The opcode is a branch (`bra`, `brx.idx`, with or without `.uni`): lanes may go elsewhere.
pub(super) fn is_branch(opcode: &str) -> bool {
    matches!(opcode.split('.').next(), Some("bra" | "brx"))
}

/// The opcode is an instruction of a later target with a plain equivalent on Volta, or with none of
/// these forms and then refused: see `Body::plain`.
pub(super) fn is_plain(opcode: &str) -> bool {
    let words: Vec<&str> = opcode.split('.').collect();
    if is_tensor(opcode) {
        return false;
    }
    opcode.starts_with("mbarrier.")
        || opcode.starts_with("cp.async.mbarrier.")
        || opcode.starts_with("griddepcontrol.")
        || opcode.starts_with("redux.sync.")
        || opcode.starts_with("createpolicy.")
        || opcode.contains(".relu.")
        || opcode.ends_with(".f32x2")
        || LATER_TYPES.iter().any(|kind| opcode.contains(kind))
        || (matches!(words[0], "ld" | "st")
            && (opcode.contains(".L2::") || opcode.contains(".volatile.local") || wide(&words).is_some()))
}

/// For a load or store of more than 128 bits: the number of registers and the bytes of each.
pub(super) fn wide(words: &[&str]) -> Option<(usize, u32)> {
    let [.., vector, kind] = words else { return None };
    let count: usize = vector.strip_prefix('v')?.parse().ok()?;
    let bytes = type_width(&format!(".{kind}"))? / 8;
    (count as u32 * bytes > 16 && count.is_multiple_of(2)).then_some((count, bytes))
}

/// The instruction `opcode operands;`, if this module has a Volta sequence for it.
/// The qualifiers of an integer product (after `mma.sync.aligned`) without `.satfinite`, and whether
/// it was there. The assembler takes it after the layouts, where the manual writes it, and after the
/// types, where the compiler writes it for Ampere's shapes; both make the same instruction, and
/// NVIDIA's libraries have both.
pub(super) fn integer_form<'a>(qualifiers: &[&'a str]) -> (Vec<&'a str>, bool) {
    match qualifiers {
        [shape, a_layout, b_layout, "satfinite", types @ ..] => {
            ([&[*shape, *a_layout, *b_layout], types].concat(), true)
        }
        [form @ .., "satfinite"] => (form.to_vec(), true),
        _ => (qualifiers.to_vec(), false),
    }
}

/// Parse a block-scaled MMA instruction: `kind::mxf4`, `kind::mxf4nvf4` or `kind::mxf8f6f4`.
///
/// Operands: `{d}, {a}, {b}, {c}, scale_a, {byte_id_a, thread_id_a}, scale_b, {byte_id_b, thread_id_b}`.
pub(super) fn parse_block_scaled<'a>(
    opcode: &str,
    qualifiers: &[&str],
    parts: &[&'a str],
    constants: &HashMap<&str, u32>,
) -> Result<Tensor<'a>, Unsupported> {
    let unsupported = |what: &str| Unsupported(format!("{opcode}: {what}"));
    // 8 operand groups: d, a, b, c, scale_a, {byte_a, thread_a}, scale_b, {byte_b, thread_b}.
    if parts.len() != 8 {
        return Err(unsupported("operands not recognised"));
    }
    let d = scan::vector(parts[0]).ok_or_else(|| unsupported("d not recognised"))?;
    let a = scan::vector(parts[1]).ok_or_else(|| unsupported("a not recognised"))?;
    let b = scan::vector(parts[2]).ok_or_else(|| unsupported("b not recognised"))?;
    let c = scan::vector(parts[3]).ok_or_else(|| unsupported("c not recognised"))?;
    let scale_a = scan::register(parts[4].trim()).ok_or_else(|| unsupported("scale_a not recognised"))?;
    // LLVM may materialize the 16-bit selectors in registers before the product.
    let selector = |value: &str| integer(value).or_else(|| constants.get(value).copied());
    let parse_selector = |part: &str| -> Option<(u32, u32)> {
        let inner = part.trim().strip_prefix('{')?.strip_suffix('}')?;
        let items: Vec<&str> = inner.split(',').collect();
        if items.len() != 2 {
            return None;
        }
        Some((selector(items[0].trim())?, selector(items[1].trim())?))
    };
    let (byte_id_a, thread_id_a) = parse_selector(parts[5]).ok_or_else(|| unsupported("selector_a not recognised"))?;
    let scale_b = scan::register(parts[6].trim()).ok_or_else(|| unsupported("scale_b not recognised"))?;
    let (byte_id_b, thread_id_b) = parse_selector(parts[7]).ok_or_else(|| unsupported("selector_b not recognised"))?;
    if d.len() != 4 || c.len() != 4 {
        return Err(unsupported("d or c length not 4"));
    }

    // Determine the kind, operand types and scale type from the qualifiers.
    let kind_qual = qualifiers.iter().find(|q| q.starts_with("kind::"));
    let scale_vec = qualifiers.iter().find(|q| q.starts_with("scale_vec::"));
    let stype = *qualifiers.last().ok_or_else(|| unsupported("missing stype"))?;

    let kind = match (kind_qual, scale_vec, stype) {
        // kind::mxf4: m16n8k64, E2M1, UE8M0, scale_vec::2X (default or explicit).
        (Some(&"kind::mxf4"), None | Some(&"scale_vec::2X"), "ue8m0") if a.len() == 4 && b.len() == 2 => {
            ScaledKind::Mxf4
        }
        // kind::mxf4nvf4 with UE8M0 scales (two or four): refused, no reference vectors from the
        // hardware exist for them and a wrong result is not announced.
        (Some(&"kind::mxf4nvf4"), _, "ue8m0") => {
            return Err(unsupported(
                "kind::mxf4nvf4 with UE8M0 scales has no Volta sequence (not verified against the hardware)",
            ));
        }
        // kind::mxf4nvf4 with UE4M3 scales: only scale_vec::4X (the assembler refuses 2X).
        (Some(&"kind::mxf4nvf4"), Some(&"scale_vec::4X"), "ue4m3") if a.len() == 4 && b.len() == 2 => {
            ScaledKind::Nvf4Ue4m3x4
        }
        // kind::mxf8f6f4: m16n8k32, E4M3/E5M2 (for now), UE8M0, scale_vec::1X.
        (Some(&"kind::mxf8f6f4"), None | Some(&"scale_vec::1X"), "ue8m0") if a.len() == 4 && b.len() == 2 => {
            // Find the operand types from the qualifiers.
            let type_quals: Vec<&str> = qualifiers.iter().filter(|q| Fp8::named(q).is_some()).copied().collect();
            match type_quals[..] {
                [a_type, b_type] => match (Fp8::named(a_type), Fp8::named(b_type)) {
                    (Some(a_fp8), Some(b_fp8)) => ScaledKind::Mxf8f6f4(a_fp8, b_fp8),
                    _ => return Err(unsupported("no Volta sequence for this form")),
                },
                _ => return Err(unsupported("no Volta sequence for this form")),
            }
        }
        _ => return Err(unsupported("no Volta sequence for this form")),
    };

    // What the selectors may be (the PTX manual's table; ptxas 13.5 refuses a byte 1 with two
    // scales): for A a pair of threads of the group, 0 or 1; for B a thread, 0 to 3;
    // the first byte of the scales: any of the four with one scale, 0 or 2 with two, 0 with four.
    // Anything else is refused here: the sequence would take bytes the register does not have.
    let byte_allowed = |byte_id: u32| match kind.blocks() {
        1 => byte_id <= 3,
        2 => byte_id == 0 || byte_id == 2,
        _ => byte_id == 0,
    };
    if !byte_allowed(byte_id_a) || !byte_allowed(byte_id_b) || thread_id_a > 1 || thread_id_b > 3 {
        return Err(unsupported("a scale selector that the instruction does not have"));
    }

    Ok(Tensor::BlockScaled {
        kind,
        d,
        a,
        b,
        c,
        scale_a,
        byte_id_a,
        thread_id_a,
        scale_b,
        byte_id_b,
        thread_id_b,
    })
}

/// A PTX floating constant, including exact single/double bit patterns. Keep the spelling:
/// ptxas applies the original conversion to f32 in the accumulator of the Volta MMA.
fn float_literal(text: &str) -> bool {
    let unsigned = text.strip_prefix(['-', '+']).unwrap_or(text);
    if let Some(bits) = unsigned.strip_prefix("0f").or_else(|| unsigned.strip_prefix("0F")) {
        return bits.len() == 8 && bits.bytes().all(|b| b.is_ascii_hexdigit());
    }
    if let Some(bits) = unsigned.strip_prefix("0d").or_else(|| unsigned.strip_prefix("0D")) {
        return bits.len() == 16 && bits.bytes().all(|b| b.is_ascii_hexdigit());
    }
    unsigned.starts_with(|c: char| c.is_ascii_digit() || c == '.') && unsigned.parse::<f64>().is_ok()
}

fn matrix_count(qualifier: &str) -> Option<usize> {
    match qualifier {
        "x1" => Some(1),
        "x2" => Some(2),
        "x4" => Some(4),
        _ => None,
    }
}

pub(super) fn parse<'a>(
    opcode: &str,
    operands: &'a str,
    constants: &HashMap<&str, u32>,
) -> Result<Tensor<'a>, Unsupported> {
    let unsupported = |what: &str| Unsupported(format!("{opcode}: {what}"));
    let parts = scan::split_operands(operands);
    let qualifiers: Vec<&str> = opcode.split('.').skip(3).collect();
    if opcode.starts_with("mma.") {
        // Block-scaled MMA: 8 operand groups (d, a, b, c, scale_a, {byte_a, thread_a},
        // scale_b, {byte_b, thread_b}). Detected from the qualifiers before the standard parse.
        if qualifiers.contains(&"block_scale") {
            return parse_block_scaled(opcode, &qualifiers, &parts, constants);
        }
        let float_accumulator = matches!(
            qualifiers[..],
            ["m16n8k8" | "m16n8k16", "row", "col", "f32", "f16", "f16", "f32"]
        );
        let vectors: Option<Vec<Vec<&str>>> = parts
            .iter()
            .enumerate()
            .map(|(i, part)| {
                if i == 3 && float_accumulator {
                    scan::vector_with(part, |item| scan::register(item).is_some() || float_literal(item))
                } else {
                    scan::vector(part)
                }
            })
            .collect();
        let (d, a, b, c) = match vectors.as_deref() {
            Some([d, a, b, c]) => (d, a, b, c),
            _ => return Err(unsupported("operands not recognised")),
        };
        let lengths = [d.len(), a.len(), b.len(), c.len()];
        let product = |half| Tensor::Product {
            half,
            d: d.clone(),
            a: a.clone(),
            b: b.clone(),
            c: c.clone(),
        };
        let sums = |element| Tensor::Sums {
            element,
            d: d.clone(),
            a: a.clone(),
            b: b.clone(),
            c: c.clone(),
        };
        match qualifiers[..] {
            ["m16n8k8", "row", "col", "f32", "f16", "f16", "f32"] if lengths == [4, 2, 1, 4] => Ok(product(false)),
            ["m16n8k8", "row", "col", "f16", "f16", "f16", "f16"] if lengths == [2, 2, 1, 2] => Ok(product(true)),
            ["m16n8k16", "row", "col", "f32", "f16", "f16", "f32"] if lengths == [4, 4, 2, 4] => Ok(product(false)),
            ["m16n8k16", "row", "col", "f16", "f16", "f16", "f16"] if lengths == [2, 4, 2, 2] => Ok(product(true)),
            ["m16n8k32", "row", "col", "f32", a_type, b_type, "f32"]
                if lengths == [4, 4, 2, 4] && Fp8::named(a_type).is_some() && Fp8::named(b_type).is_some() =>
            {
                match (Fp8::named(a_type), Fp8::named(b_type)) {
                    (Some(a_type), Some(b_type)) => Ok(Tensor::Fp8Product {
                        a_type,
                        b_type,
                        d: d.clone(),
                        a: a.clone(),
                        b: b.clone(),
                        c: c.clone(),
                    }),
                    _ => Err(unsupported("no Volta sequence for this form")),
                }
            }
            ["m16n8k16", "row", "col", "f32", "bf16", "bf16", "f32"] if lengths == [4, 4, 2, 4] => {
                Ok(sums(Element::Bf16))
            }
            ["m16n8k8", "row", "col", "f32", "bf16", "bf16", "f32"] if lengths == [4, 2, 1, 4] => {
                Ok(sums(Element::Bf16))
            }
            ["m16n8k8", "row", "col", "f32", "tf32", "tf32", "f32"] if lengths == [4, 4, 2, 4] => {
                Ok(sums(Element::Tf32))
            }
            ["m16n8k4", "row", "col", "f32", "tf32", "tf32", "f32"] if lengths == [4, 2, 1, 4] => {
                Ok(sums(Element::Tf32))
            }
            ["m8n8k4", "row", "col", "f64", "f64", "f64", "f64"] if lengths == [2, 1, 1, 2] => Ok(sums(Element::F64)),
            // `satfinite` taken out: see `integer_form`.
            _ => {
                let (form, satfinite) = integer_form(&qualifiers);
                let dot = |element: &str| match element {
                    "s8" => Some("s32"),
                    "u8" => Some("u32"),
                    _ => None,
                };
                let [shape, "row", "col", "s32", a_type, b_type, "s32"] = form[..] else {
                    return Err(unsupported("no Volta sequence for this form"));
                };
                let ampere = match (shape, lengths) {
                    ("m8n8k16", [2, 1, 1, 2]) => false,
                    ("m16n8k16", [4, 2, 1, 4]) | ("m16n8k32", [4, 4, 2, 4]) => true,
                    _ => return Err(unsupported("no Volta sequence for this form")),
                };
                let (Some(a_type), Some(b_type)) = (dot(a_type), dot(b_type)) else {
                    return Err(unsupported("element types not supported"));
                };
                let types = format!("{a_type}.{b_type}");
                Ok(if ampere {
                    Tensor::AmpereBytes {
                        types,
                        satfinite,
                        d: d.clone(),
                        a: a.clone(),
                        b: b.clone(),
                        c: c.clone(),
                    }
                } else {
                    Tensor::Bytes {
                        types,
                        satfinite,
                        d: d.clone(),
                        a: a[0],
                        b: b[0],
                        c: c.clone(),
                    }
                })
            }
        }
    } else if opcode.starts_with("stmatrix.sync.aligned.") {
        let (count, transposed) = match qualifiers[..] {
            ["m8n8", count, "shared" | "shared::cta", "b16"] => (count, false),
            ["m8n8", count, "trans", "shared" | "shared::cta", "b16"] => (count, true),
            _ => return Err(unsupported("no Volta sequence for this form")),
        };
        let count = matrix_count(count).ok_or_else(|| unsupported("matrix count not recognised"))?;
        if let [place, vector] = parts.as_slice()
            && let (Some((base, offset)), Some(src)) = (address(place), scan::vector(vector))
            && src.len() == count
        {
            Ok(Tensor::Store {
                transposed,
                src,
                base,
                offset,
            })
        } else {
            Err(unsupported("operands not recognised"))
        }
    } else if opcode.starts_with("ldmatrix.") {
        // The shape is written first (`m8n8.x4.trans`) or after the count and `trans` (`x4.trans.m8n8`).
        let mut qualifiers = qualifiers;
        if let Some(at) = qualifiers.iter().position(|qualifier| *qualifier == "m8n8") {
            qualifiers[..=at].rotate_right(1);
        }
        // nvcc also emits trans.m8n8.x4; after moving the shape first the count follows trans.
        if matches!(qualifiers[..], ["m8n8", "trans", "x1" | "x2" | "x4", ..]) {
            qualifiers.swap(1, 2);
        }
        let (count, transposed, shared) = match qualifiers[..] {
            ["m8n8", count, "b16"] => (count, false, false),
            ["m8n8", count, "trans", "b16"] => (count, true, false),
            ["m8n8", count, "shared" | "shared::cta", "b16"] => (count, false, true),
            ["m8n8", count, "trans", "shared" | "shared::cta", "b16"] => (count, true, true),
            _ => return Err(unsupported("no Volta sequence for this form")),
        };
        let count = matrix_count(count).ok_or_else(|| unsupported("matrix count not recognised"))?;
        match parts[..] {
            [r, p] => match (scan::vector(r), address(p)) {
                (Some(r), Some((base, offset))) if r.len() == count => Ok(Tensor::Load {
                    transposed,
                    shared,
                    r,
                    base,
                    offset,
                }),
                _ => Err(unsupported("operands not recognised")),
            },
            _ => Err(unsupported("operands not recognised")),
        }
    } else if opcode == "movmatrix.sync.aligned.m8n8.trans.b16" {
        match parts[..] {
            [d, a] if scan::register(d).is_some() && scan::register(a).is_some() => Ok(Tensor::Transpose { d, a }),
            _ => Err(unsupported("operands not recognised")),
        }
    } else if matches!(
        opcode,
        "cp.async.commit_group" | "cp.async.wait_group" | "cp.async.wait_all"
    ) {
        Ok(Tensor::Copied)
    } else if opcode.starts_with("cp.async.") {
        // cp.async.{ca,cg}.shared{::cta}.global{.L2::<prefetch size>}: the prefetch size is a hint.
        let words: Vec<&str> = opcode.split('.').collect();
        let plain = matches!(words[2..], [_, "shared" | "shared::cta", "global", ref hints @ ..]
            if hints.iter().all(|hint| matches!(*hint, "L2::64B" | "L2::128B" | "L2::256B")));
        let (to, from, size, amount) = match parts[..] {
            [to, from, size] => (to, from, size, None),
            [to, from, size, amount] => (to, from, size, Some(amount)),
            _ => return Err(unsupported("operands not recognised")),
        };
        match (words[2], plain, place_of(to), place_of(from), integer(size)) {
            ("ca", true, Some(to), Some(from), Some(size @ (4 | 8 | 16)))
            | ("cg", true, Some(to), Some(from), Some(size @ 16)) => Ok(Tensor::Copy { size, to, from, amount }),
            _ => Err(unsupported("no Volta sequence for this form")),
        }
    } else if let Some(plain) = match opcode {
        "min.NaN.f32" => Some("min.f32"),
        "max.NaN.f32" => Some("max.f32"),
        "min.ftz.NaN.f32" => Some("min.ftz.f32"),
        "max.ftz.NaN.f32" => Some("max.ftz.f32"),
        _ => None,
    } {
        match parts[..] {
            [d, a, b] if scan::register(d).is_some() => Ok(Tensor::Bound {
                plain,
                d,
                a,
                b,
                third: None,
            }),
            [d, a, b, c] if scan::register(d).is_some() => Ok(Tensor::Bound {
                plain,
                d,
                a,
                b,
                third: Some(c),
            }),
            _ => Err(unsupported("operands not recognised")),
        }
    } else if let Some(rounding) = match opcode {
        "cvt.rn.f16x2.f32" => Some("rn"),
        "cvt.rz.f16x2.f32" => Some("rz"),
        _ => None,
    } {
        match parts[..] {
            [d, a, b] if [d, a, b].iter().all(|operand| scan::register(operand).is_some()) => {
                Ok(Tensor::Pack { rounding, d, a, b })
            }
            _ => Err(unsupported("operands not recognised")),
        }
    } else {
        Err(unsupported("no Volta sequence for this form"))
    }
}
