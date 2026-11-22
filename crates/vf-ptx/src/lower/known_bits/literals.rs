//! Find literal register values that survive labels or unrelated instructions.
use super::word_literal;
use crate::scan::{self, Item, Kind};
use std::collections::HashMap;

pub(super) fn immutable<'a>(text: &'a str, items: &[Item]) -> HashMap<&'a str, u32> {
    let mut labels = HashMap::new();
    for (i, item) in items.iter().enumerate().filter(|(_, it)| it.kind == Kind::Label) {
        let label = text[item.range.clone()].trim_end_matches(':').trim();
        if labels.insert(label, i).is_some() {
            return HashMap::new();
        }
    }
    // A literal in the entry region dominates all subsequent execution if every
    // earlier jump lands before its definition. Other literals are local to the
    // defining basic block. In either case, verify every syntactic use below.
    let mut jump_target = 0;
    let mut literals = HashMap::new();
    for (i, item) in items.iter().enumerate() {
        let Kind::Statement {
            guarded,
            opcode,
            operands,
        } = &item.kind
        else {
            continue;
        };
        let (op, args) = (&text[opcode.clone()], &text[operands.clone()]);
        if matches!(op, "mov.b32" | "mov.s32" | "mov.u32")
            && !guarded
            && let [name, n] = scan::split_operands(args).as_slice()
            && scan::register(name).is_some()
            && let Some(bits) = word_literal(n)
        {
            literals.insert(*name, (bits, i, jump_target <= i));
        }
        match op {
            "bra" | "bra.uni" => jump_target = jump_target.max(*labels.get(args.trim()).unwrap_or(&usize::MAX)),
            _ if matches!(
                op.split('.').next(),
                Some(
                    // Straight-line prologue operations. Other instructions may transfer control.
                    "" | "add"
                        | "sub"
                        | "mul"
                        | "mad"
                        | "div"
                        | "rem"
                        | "neg"
                        | "abs"
                        | "cvt"
                        | "cvta"
                        | "and"
                        | "or"
                        | "xor"
                        | "not"
                        | "shl"
                        | "shr"
                        | "mov"
                        | "min"
                        | "max"
                        | "setp"
                        | "selp"
                        | "ld"
                        | "st"
                )
            ) => {}
            _ => jump_target = usize::MAX,
        }
    }
    let mut last_label = 0;
    for (i, item) in items.iter().enumerate() {
        let (op, args) = match &item.kind {
            Kind::Label => {
                last_label = i;
                continue;
            }
            Kind::Statement { opcode, operands, .. } => (&text[opcode.clone()], &text[operands.clone()]),
            _ => continue,
        };
        if op == ".reg" {
            continue;
        } // The caller excludes every possibly shadowed name.
        let reads = matches!(op, "and.b32" | "or.b32" | "shr.u32" | "fma.rn.bf16x2");
        let parts = scan::split_operands(args);
        for name in scan::names(args) {
            if let Some(&(_, def, entry)) = literals.get(name)
                && i != def
                && !(i > def
                    && (entry || last_label <= def)
                    && reads
                    && parts.first().is_some_and(|d| !scan::names(d).any(|dest| dest == name))
                    && parts[1..].contains(&name))
            {
                literals.remove(name);
            }
        }
    }
    literals.into_iter().map(|(name, (bits, _, _))| (name, bits)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn literal(source: &str) -> Option<u32> {
        let body = format!("{{{source}}}");
        let items = scan::items(body.as_bytes(), 0..body.len()).unwrap();
        immutable(&body, &items).get("%m").copied()
    }

    #[test]
    fn only_read_only_literals_that_reach_every_use_are_kept() {
        let def = "mov.b32 %m,0x000f000f;";
        let use_it = "and.b32 %d,%v,%m;";
        for body in [
            format!("{def} @%p bra end; end: {use_it}"),
            format!("@%p bra early; early: {def} @%p bra end; end: {use_it}"),
            format!("@%p bra end; {def} ld.shared.u32 %x,[%a]; {use_it} end:"),
        ] {
            assert_eq!(literal(&body), Some(0x000f000f), "{body}");
        }
        for body in [
            format!("@%p bra join; {def} join: {use_it}"),
            format!("@%p {def} {use_it}"),
            format!("{def} @%p mov.b32 %m,%x; {use_it}"),
            format!("{use_it} {def} {use_it}"),
            format!("{def} opaque %x,%m; {use_it}"),
            format!("brx.idx %idx,jumps; {def} join: {use_it}"),
            format!("{def} or.b32 %m,%x,%m; {use_it}"),
        ] {
            assert_eq!(literal(&body), None, "{body}");
        }
    }
}
