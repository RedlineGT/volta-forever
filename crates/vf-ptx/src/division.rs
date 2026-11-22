//! Version short, straight-line loops with repeated signed division by a kernel argument.
//! On Volta a register divide expands to many instructions. A uniform guard outside
//! the loop lets positive powers of two use a shift with truncation towards zero.

use crate::scan::{self, Item, Kind};
use std::fmt::Write as _;
use std::ops::Range;

const PREFIX: &str = "__vf_pow2div_";
const MIN_DIVISIONS: usize = 4;
const MAX_LOOP_ITEMS: usize = 192;

fn instruction<'a>(text: &'a str, item: &Item) -> Option<(bool, &'a str, &'a str)> {
    let Kind::Statement {
        guarded,
        opcode,
        operands,
    } = &item.kind
    else {
        return None;
    };
    Some((*guarded, &text[opcode.clone()], text[operands.clone()].trim()))
}

fn division<'a>(text: &'a str, item: &Item) -> Option<[&'a str; 3]> {
    let (false, "div.s32", args) = instruction(text, item)? else {
        return None;
    };
    let regs: [&str; 3] = scan::split_operands(args).try_into().ok()?;
    regs.iter()
        .all(|r| r.starts_with('%') && scan::register(r).is_some())
        .then_some(regs)
}

fn alternate_number(name: &str) -> bool {
    let tail = name.rsplit_once(|c: char| !c.is_ascii_digit()).map_or(name, |(_, n)| n);
    tail.len() > 1 && tail.starts_with('0')
}

fn argument(signature: &str, address: &str) -> bool {
    let Some(name) = address
        .trim()
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .map(str::trim)
    else {
        return false;
    };
    let Some((_, after)) = signature.split_once('(') else {
        return false;
    };
    let Some((params, _)) = after.split_once(')') else {
        return false;
    };
    scan::register(name).is_some()
        && !name.starts_with('%')
        && params.split(',').any(|param| {
            let mut words = param.split_whitespace();
            words.next() == Some(".param")
                && matches!(words.next(), Some(".b32" | ".s32" | ".u32"))
                && words.next() == Some(name)
                && words.next().is_none()
        })
}

// Require an unconditional load in the entry block and no other definitions of the
// divisor. PTX permits bare register names for indirect ld.param; restricting declarations
// to %-prefixed registers keeps their addresses distinct from the named scalar argument.
fn uniform_divisor(text: &str, items: &[Item], signature: &str, divisor: &str) -> bool {
    let mut loaded = false;
    let mut entry = true;
    for item in items {
        let Some((guarded, op, args)) = instruction(text, item) else {
            if item.kind != Kind::Label {
                return false;
            }
            entry = false;
            continue;
        };
        if op == ".reg" {
            let Some((_, names)) = args.split_once(char::is_whitespace) else {
                return false;
            };
            if !names.split(',').all(|name| name.trim().starts_with('%')) {
                return false;
            }
            continue;
        }
        if op == ".param" {
            return false;
        }
        if guarded || matches!(op.split('.').next(), Some("bra" | "brx" | "call" | "ret" | "exit")) {
            entry = false;
        }
        if !scan::names(args).any(|r| r == divisor) {
            continue;
        }
        if !loaded
            && entry
            && matches!(op, "ld.param.b32" | "ld.param.s32" | "ld.param.u32")
            && let Some((dst, address)) = args.split_once(',')
            && dst.trim() == divisor
            && argument(signature, address)
        {
            loaded = true;
        } else if !loaded || !division(text, item).is_some_and(|[out, _, d]| out != divisor && d == divisor) {
            return false;
        }
    }
    loaded
}

fn duplicable(op: &str, args: &str) -> bool {
    if op == ".pragma" {
        return args == "\"nounroll\"";
    }
    matches!(
        op.split('.').next(),
        Some(
            "ld" | "st" | "mov" | "add" | "sub" | "mul" | "mad" | "shl" | "shr" | "and" | "or" | "xor" | "setp" | "cvt"
        )
    ) && !op.contains("volatile")
        && !op.contains("acquire")
        && !op.contains("release")
}

struct Quotient<'a> {
    range: Range<usize>,
    result: &'a str,
    numerator: &'a str,
}

fn render(source: &str, head: &Item, back: &Item, divisor: &str, quotients: &[Quotient<'_>]) -> String {
    let original = &source[head.range.start..back.range.end];
    let mut result = format!(
        ".reg .pred %__vf_pow2div_p;
.reg .b32 %__vf_pow2div_mask,%__vf_pow2div_bits,%__vf_pow2div_shift,%__vf_pow2div_sign,%__vf_pow2div_adj;
sub.u32 %__vf_pow2div_mask,{divisor},1;
and.b32 %__vf_pow2div_bits,%__vf_pow2div_mask,{divisor};
setp.eq.u32 %__vf_pow2div_p,%__vf_pow2div_bits,0;
setp.gt.and.s32 %__vf_pow2div_p,{divisor},0,%__vf_pow2div_p;
bfind.u32 %__vf_pow2div_shift,{divisor};
@%__vf_pow2div_p bra $__vf_pow2div_fast;
{original}
bra $__vf_pow2div_done;
$__vf_pow2div_fast:"
    );
    let mut cursor = head.range.end;
    // (n + ((n >> 31) & (divisor - 1))) >> log2(divisor) also handles INT_MIN.
    for Quotient {
        range,
        result: dest,
        numerator: n,
    } in quotients
    {
        result.push_str(&source[cursor..range.start]);
        let _ = write!(
            result,
            "shr.s32 %__vf_pow2div_sign,{n},31;
and.b32 %__vf_pow2div_adj,%__vf_pow2div_sign,%__vf_pow2div_mask;
add.s32 %__vf_pow2div_adj,{n},%__vf_pow2div_adj;
shr.s32 {dest},%__vf_pow2div_adj,%__vf_pow2div_shift;"
        );
        cursor = range.end;
    }
    let Kind::Statement { opcode, .. } = &back.kind else {
        unreachable!()
    };
    result.push_str(&source[cursor..opcode.end]);
    result.push_str(" $__vf_pow2div_fast;\n$__vf_pow2div_done:\n");
    result
}

fn specialized(source: &str, text: &str, f: &crate::Function) -> Option<(Range<usize>, String)> {
    let body = &text[f.body.clone()];
    if body.match_indices("div.s32").take(MIN_DIVISIONS).count() < MIN_DIVISIONS {
        return None;
    }
    // A family member written %r01 aliases %r1. Keep such kernels as supplied.
    if scan::names(body).any(|n| n.starts_with('%') && alternate_number(n)) {
        return None;
    }
    let items = scan::items(text.as_bytes(), f.body.clone())?;
    let signature = &text[f.name.end..f.body.start];
    for (index, head) in items.iter().enumerate().filter(|(_, i)| i.kind == Kind::Label) {
        let label = text[head.range.clone()].trim_end_matches(':').trim();
        let mut quotients = Vec::new();
        let mut divisor = None;
        for item in items.iter().skip(index + 1).take(MAX_LOOP_ITEMS) {
            let Some((guarded, op, args)) = instruction(text, item) else {
                break;
            };
            if guarded && matches!(op, "bra" | "bra.uni") && args == label {
                if quotients.len() >= MIN_DIVISIONS
                    && let Some(d) = divisor
                    && uniform_divisor(text, &items, signature, d)
                {
                    return Some((
                        head.range.start..item.range.end,
                        render(source, head, item, d, &quotients),
                    ));
                }
                break;
            }
            if let Some([result, numerator, d]) = division(text, item) {
                if divisor.is_some_and(|old| old != d) {
                    break;
                }
                divisor = Some(d);
                quotients.push(Quotient {
                    range: item.range.clone(),
                    result,
                    numerator,
                });
            } else if guarded || !duplicable(op, args) {
                break;
            }
        }
    }
    None
}

/// Specialize one eligible loop per entry. Never rewrite a native image or device
/// function, or shadow an identifier supplied by the program.
pub fn specialize(ptx: &mut Vec<u8>) {
    if !ptx.windows(7).any(|w| w == b"div.s32") {
        return;
    }
    let Ok(source) = std::str::from_utf8(ptx) else { return };
    if source.contains(PREFIX) {
        return;
    }
    let text = scan::blank_comments(source);
    let edits: Vec<_> = crate::functions(text.as_bytes())
        .into_iter()
        .filter(|f| f.entry)
        .filter_map(|f| specialized(source, &text, &f))
        .collect();
    for (range, replacement) in edits.into_iter().rev() {
        ptx.splice(range, replacement.bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = ".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry q(.param .u32 divisor) {
.reg .b32 %r<10>; .reg .pred %p;
ld.param.u32 %r0,[divisor];
L:
div.s32 %r1,%r2,%r0; div.s32 %r2,%r3,%r0; div.s32 %r3,%r4,%r0; div.s32 %r4,%r1,%r0;
@%p bra L;
ret;
}";

    fn adapt(s: &str) -> String {
        let mut text = s.as_bytes().to_vec();
        specialize(&mut text);
        String::from_utf8(text).unwrap()
    }

    #[test]
    fn uniform_divisor_versions_only_the_backedge_and_preserves_the_original_loop() {
        let text = FIXTURE.replace("bra L;", "bra /* L */ L /* L */;");
        let result = adapt(&text);
        assert!(result.contains(&text[text.find("L:").unwrap()..text.find("\nret;").unwrap()]));
        assert!(result.contains("@%p bra $__vf_pow2div_fast;"));
        assert_eq!(result.matches("shr.s32 %__vf_pow2div_sign,").count(), 4);
        assert_eq!(adapt(&result), result);
    }

    #[test]
    fn unsupported_control_flow_names_and_writes_leave_the_kernel_intact() {
        for text in [
            FIXTURE.replace("L:\n", "L:\nmov.u32/*adjacent*/%r0,3;\n"),
            FIXTURE.replace("L:\n", "L:\nmov.u32 %r00,3;\n"),
            FIXTURE.replace("L:\n", "L:\n@%p mov.u32 %r0,3;\n"),
            FIXTURE.replace("L:\n", "L:\nbar.sync 0;\n"),
            FIXTURE.replace("L:\n", "L:\n.loc 1 3 0 mov.u32 %r0,3;\n"),
            FIXTURE.replace("L:\n", "L:\n{ .reg .b32 %r0; }\n"),
            FIXTURE.replace("ld.param", "@%p ld.param"),
            FIXTURE
                .replace("ld.param", "bra after; ld.param")
                .replace("L:", "after: L:"),
            FIXTURE
                .replace("[divisor]", "[address]")
                .replace("ld.param", ".reg .u64 address; ld.param"),
            FIXTURE.replace("[divisor]", "[absent]"),
            FIXTURE.replace("%r4,%r1,%r0", "%r4,%r1,%r5"),
            FIXTURE.replace("%r4,%r1,%r0", "%r0,%r1,%r0"),
            FIXTURE
                .replace("L:", "L__:")
                .replace("bra L;", "bra L__; // __vf_pow2div_p"),
            FIXTURE.replace(".entry", ".func"),
        ] {
            assert_eq!(adapt(&text), text, "{text}");
        }
    }
}
