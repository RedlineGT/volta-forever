//! Kernels that a module holds without code: a body that is a trap and nothing else, or a body that
//! does nothing.
//!
//! A compiler that builds a kernel for some architectures only still emits it for the others, with
//! a body that traps (CUDA's `__trap()` in the `#else` of an architecture test) or with an empty one.
//! Such a kernel is not code that the layer could run otherwise: launched on any GPU, the first
//! traps and the GPU ends the context, the second does nothing. The layer does not refuse them at a
//! lookup (fat binaries are full of them, and programs that never launch them look them up all the
//! same); it refuses a launch of the first (`vf-proxy`, `guard`), and names the second at log
//! level 2.

use crate::functions;
use crate::scan::{self, Kind};

/// What the instructions of a body come to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Body {
    /// Its first instruction is a `trap` without a guard, and the others only return: it traps
    /// whenever it runs, before anything else.
    TrapOnly,
    /// It has no instruction but returns.
    Empty,
    /// Anything else, or a body that does not divide into statements.
    Code,
}

/// What `text[body]` (a function body, braces included) does. Declarations and directives
/// (`.reg`, `.local`, `.pragma`, ...), debug lines, labels and inner braces are not instructions.
/// Strict on purpose: a `trap` under a guard, after a branch or after any other instruction is code.
fn classify(text: &[u8], body: std::ops::Range<usize>) -> Body {
    // A body of more statements than this is code: the test reads every body of modules of tens of
    // megabytes, and a body without code is a few statements (and a few declarations).
    const AT_MOST: usize = 32;
    if text[body.clone()].iter().filter(|&&b| b == b';').nth(AT_MOST).is_some() {
        return Body::Code;
    }
    let Some(items) = scan::items(text, body) else {
        return Body::Code;
    };
    let mut first = None;
    for item in &items {
        let Kind::Statement { guarded, opcode, .. } = &item.kind else {
            continue;
        };
        let opcode = &text[opcode.clone()];
        if opcode.first() == Some(&b'.') {
            continue;
        }
        if *guarded {
            return Body::Code;
        }
        let base = opcode.split(|&b| b == b'.').next().unwrap_or(opcode);
        match base {
            b"trap" if first.is_none() => first = Some(Body::TrapOnly),
            b"ret" | b"exit" => {
                first.get_or_insert(Body::Empty);
            }
            _ => return Body::Code,
        }
    }
    first.unwrap_or(Body::Empty)
}

/// The kernels of a module without code, by what their body is.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Stubs {
    /// The kernels (`.entry`) whose body is a trap and nothing else: the first instruction is an
    /// unguarded `trap`, and the only others are `ret` and `exit`. Launched, such a kernel ends the
    /// context on any GPU. Device functions are not looked at: a function that only traps is how a
    /// program aborts, and a kernel that calls it does other things too.
    pub trap_only: Vec<String>,
    /// The kernels whose body has no instruction but returns: launched, they do nothing. (Some are
    /// meant to: a kernel launched to measure a launch, or to load a module.)
    pub empty: Vec<String>,
}

/// The kernels of the module `text` without code (`Stubs`).
pub fn stubs(text: &[u8]) -> Stubs {
    let mut found = Stubs::default();
    for function in functions(text).into_iter().filter(|function| function.entry) {
        let list = match classify(text, function.body.clone()) {
            Body::TrapOnly => &mut found.trap_only,
            Body::Empty => &mut found.empty,
            Body::Code => continue,
        };
        list.push(String::from_utf8_lossy(&text[function.name]).into_owned());
    }
    found
}

/// `stubs(text).trap_only`.
pub fn trap_only_kernels(text: &[u8]) -> Vec<String> {
    stubs(text).trap_only
}

/// `stubs(text).empty`.
pub fn empty_kernels(text: &[u8]) -> Vec<String> {
    stubs(text).empty
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(bodies: &[(&str, &str)]) -> String {
        let mut text = String::from(".version 9.0\n.target sm_120\n.address_size 64\n\n");
        for (name, body) in bodies {
            text.push_str(&format!(
                ".visible .entry {name}(\n.param .u64 {name}_p\n)\n.maxntid 256\n{{\n{body}\n}}\n\n"
            ));
        }
        text
    }

    #[test]
    fn a_body_that_only_traps_is_found() {
        // What nvcc 13 writes for an instantiation built without code for the target: `trap; ret;`.
        let text = module(&[
            ("stub", "\n\n\ntrap;\n\nret;\n"),
            ("stub_exit", "trap;\nexit;"),
            ("stub_alone", "trap;"),
            // Declarations, a debug line, a label, comments and inner braces are not instructions.
            (
                "stub_dressed",
                ".reg .b32 %r<2>;\n.loc 1 7 3\n// a comment\n$L__BB0_1:\n{\n/* inner */ trap;\n}\nret.uni;",
            ),
        ]);
        assert_eq!(
            trap_only_kernels(text.as_bytes()),
            ["stub", "stub_exit", "stub_alone", "stub_dressed"]
        );
        assert!(empty_kernels(text.as_bytes()).is_empty());
    }

    #[test]
    fn a_trap_after_an_effect_or_behind_a_branch_is_code() {
        let text = module(&[
            // An effect first: the store happens before the trap.
            (
                "store_then_trap",
                ".reg .b64 %rd<2>;\nld.param.u64 %rd1, [store_then_trap_p];\nst.global.u32 [%rd1], 1;\ntrap;\nret;",
            ),
            // Under a guard: it traps only when the predicate holds.
            ("guarded", ".reg .pred %p<2>;\nsetp.eq.u32 %p1, 1, 1;\n@%p1 trap;\nret;"),
            ("guarded_alone", "@%p1 trap;\nret;"),
            ("negated", "@!%p1 trap;\nret;"),
            // Behind a branch.
            ("branch", "bra $L__end;\ntrap;\n$L__end:\nret;"),
            // Returns first: the trap is never reached.
            ("returns_first", "ret;\ntrap;"),
            // A call that might not return.
            ("call_then_trap", "call.uni abort, ();\ntrap;"),
        ]);
        assert!(trap_only_kernels(text.as_bytes()).is_empty());
        assert!(empty_kernels(text.as_bytes()).is_empty());
    }

    #[test]
    fn an_empty_body_is_told_apart() {
        let text = module(&[
            ("noop", "\n\nret;\n"),
            ("bare", ""),
            ("only_decls", ".reg .b32 %r<3>;\nret;"),
        ]);
        assert_eq!(empty_kernels(text.as_bytes()), ["noop", "bare", "only_decls"]);
        assert!(trap_only_kernels(text.as_bytes()).is_empty());
    }

    #[test]
    fn a_device_function_that_only_traps_is_not_a_kernel_without_code() {
        let text = ".version 9.0\n.target sm_75\n.address_size 64\n\n.func abort()\n{\ntrap;\nret;\n}\n\n\
                    .visible .entry k()\n{\ncall.uni abort, ();\nret;\n}\n";
        assert!(trap_only_kernels(text.as_bytes()).is_empty());
    }

    #[test]
    fn a_body_that_does_not_divide_into_statements_is_code() {
        let text = module(&[("broken", "trap")]);
        assert!(trap_only_kernels(text.as_bytes()).is_empty());
    }
}
