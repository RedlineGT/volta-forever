//! Dependency analysis for PTX modules: which symbols each function references, and extraction of
//! sub-modules containing a single entry function with its transitive dependencies.
//!
//! A PTX module has:
//! - Module-level directives: `.version`, `.target`, `.address_size`
//! - Module-level declarations and definitions: `.global`, `.const`, `.shared`, `.extern`, etc.
//! - Function declarations (`.func ... ;`) and definitions (`.func ... { ... }`, `.entry ... { ... }`)
//!
//! A sub-module for one entry function contains the module header, the entry function, every device
//! function it calls transitively, every global/const/shared symbol any of them references, and the
//! declarations of every function they call that is not defined in this module.
//!
//! Splitting is unsafe when two entry functions in the same module write to the same module-level
//! variable (the value written by one kernel would not be seen by the other if they are in separate
//! driver modules). The analysis is conservative: if it cannot determine that splitting is safe, the
//! module is loaded whole.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use crate::scan;

mod calls;

fn is_name_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'$' | b'%' | b'.' | b'@')
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Linkage {
    Private,
    Weak,
    DynamicShared,
    External,
}

/// A module-level variable declaration or definition.
#[derive(Debug, Clone, PartialEq, Eq)]
struct VarDecl {
    /// Byte range in the module text (the complete declaration, including the trailing newline).
    range: Range<usize>,
    /// The identifier name.
    name: String,
    linkage: Linkage,
    /// Whether this is a `.global` or `.const` variable (persists across launches). `.shared` is
    /// per-block per-launch and does not cause coherence issues when split.
    persistent: bool,
    /// Whether its bytes are all given by the module, which keeps its name to itself: a definition
    /// of plain data with an initializer and no linkage directive (not `.visible`, `.weak`,
    /// `.extern` or `.common`: a name that other translation units may use for shared state), not
    /// managed memory, not a texture, sampler or surface reference. Every copy of it starts with the
    /// same bytes.
    initialized_data: bool,
}

/// A module-level function declaration (no body).
#[derive(Debug, Clone, PartialEq, Eq)]
struct FuncDecl {
    range: Range<usize>,
    name: String,
}

/// An entry-point or device-function definition.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FuncDef {
    /// Range of the entire definition (from the keyword to the closing `}`).
    range: Range<usize>,
    /// The function name.
    name: String,
    /// Range of the body (between the braces, inclusive).
    body: Range<usize>,
    /// Range from the .func or .entry keyword to the opening brace.
    signature: Range<usize>,
    /// Is this an `.entry` (kernel)?
    is_entry: bool,
}

/// The dependency structure of a PTX module.
#[derive(Debug, PartialEq, Eq)]
pub struct ModuleInfo {
    /// The full PTX text.
    text: Vec<u8>,
    /// Byte range of the module header (`.version`, `.target`, `.address_size`, leading comments).
    header: Range<usize>,
    /// `.file` directives at module level (kept in every sub-module for debug info).
    file_directives: Vec<Range<usize>>,
    /// Module-level variable declarations.
    variables: Vec<VarDecl>,
    /// Module-level function declarations (extern or forward, no body).
    func_decls: Vec<FuncDecl>,
    /// Function definitions (entries and device functions).
    func_defs: Vec<FuncDef>,
    /// Entry function names, in order.
    pub entries: Vec<String>,
    /// For each function (by name), functions it references in its body or through variable initializers.
    func_refs: HashMap<String, HashSet<String>>,
    /// Functions containing a call through a prototype, whose runtime target is unknown.
    unbounded_calls: HashSet<String>,
    /// For each function, module variables it references in its body or through initializers.
    var_refs: HashMap<String, HashSet<String>>,
    /// Module-level variable names that are written (stored to) by at least one function.
    written_vars: HashSet<String>,
    /// Variables whose address escapes through another variable's initializer. Treated as written
    /// for the split rule: the address could be loaded by any kernel and stored through.
    escaped_vars: HashSet<String>,
    /// Persistent variables that any function loads data from (the variable's name or a register
    /// tainted by it appears in the address operand of a load instruction).
    loaded_vars: HashSet<String>,
    /// Persistent variables that kernels can only read, and whose bytes the module gives: each
    /// sub-module can have a copy of its own, which holds what the others hold for as long as the
    /// program does not write one (it cannot without asking for its address by name).
    read_only_vars: HashSet<String>,
    /// Names of persistent (.global, .const) variables.
    persistent_names: HashSet<String>,
    /// True when the module contains text at module level that the parser does not recognise.
    pub has_unrecognised_items: bool,
    /// File-scope pragmas can alter every function in the module.
    has_module_pragma: bool,
}

/// Whether a module can be split into per-kernel sub-modules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Splittable {
    /// Safe to split.
    Yes,
    /// Not safe to split, with the reason.
    No { reason: String },
}

// ---------------------------------------------------------------------------
// Body analysis
// ---------------------------------------------------------------------------

/// Find references to declared names, skipping numbers, comments and strings. A name beginning
/// with % may denote a user-defined symbol as well as a register.
fn find_referenced_names(text: &str, declared: impl Fn(&str) -> bool) -> HashSet<&str> {
    let bytes = text.as_bytes();
    let mut names = HashSet::new();
    let mut i = 0;
    while i < bytes.len() {
        if let Some(next) = scan::skip_comment(bytes, i) {
            i = next;
            continue;
        }
        if bytes[i] == b'"' {
            i += 1;
            while i < bytes.len() {
                match bytes[i] {
                    b'\\' => i += 2,
                    b'"' => {
                        i += 1;
                        break;
                    }
                    _ => i += 1,
                }
            }
            continue;
        }
        if bytes[i].is_ascii_digit() {
            while i < bytes.len() && is_name_byte(bytes[i]) {
                i += 1;
            }
            continue;
        }
        if matches!(bytes[i], b'_' | b'$' | b'%') || bytes[i].is_ascii_alphabetic() {
            let start = i;
            while i < bytes.len() && is_name_byte(bytes[i]) {
                i += 1;
            }
            // nvcc often uses $L for labels, but a declared symbol can have that prefix too.
            let name = &text[start..i];
            if declared(name) {
                names.insert(name);
            }
            continue;
        }
        i += 1;
    }
    names
}

/// Flow-insensitive taint analysis: which module-level variables might be written through a pointer
/// in this function body.
///
/// Uses `scan::items` for statement parsing (handles braces, multi-line statements, inline asm).
/// A register is tainted by a variable when any instruction has the variable's name or a tainted
/// register among its source operands. Loads through a tainted address give data, not taint.
/// The variable counts as written when a tainted register appears in a store, atomic, reduction,
/// or call instruction. Iterates to true fixed point (the register sets are finite).
///
/// Performance: builds a reverse map from names to the variables they could taint, so the inner
/// loop is proportional to (statements x distinct names per statement), not (statements x variables).
/// CUDA runtime device functions that only read, not write through, their pointer arguments.
/// Safe to exclude from the taint-to-write rule only when declared `.extern .func` (not defined
/// in the module).
const READONLY_EXTERN_CALLEES: &[&str] = &["vprintf", "__assertfail"];

/// The instructions through which an address only becomes another value: what they give is tainted
/// by what they are given, and they read and write no memory (by the opcode's first word). Not a
/// comparison (`setp`), nor an instruction that sets the carry (`.cc`): the taint follows what
/// instructions give, not a predicate that guards or branches, nor the carry, and an address could
/// be rebuilt bit by bit from those. (No kernel of Strata's CUDA 13 engine 0.1.64 or of
/// llama.cpp b12418 that reads a table compares its address or carries from it, 2026-11-23.)
const ADDRESS_ARITHMETIC: &[&str] = &[
    "mov", "cvta", "cvt", "add", "sub", "mul", "mad", "mul24", "mad24", "and", "or", "xor", "not", "shl", "shr", "shf",
    "selp", "prmt", "min", "max", "neg", "bfe", "bfi", "lop3",
];

/// Whether an instruction only computes a value from what it is given (`ADDRESS_ARITHMETIC`).
fn is_address_arithmetic(opcode: &str) -> bool {
    let mut words = opcode.split('.');
    ADDRESS_ARITHMETIC.contains(&words.next().unwrap_or_default()) && !words.any(|word| word == "cc")
}

/// What the taint analysis of a function body found: the variables written, loaded from, and not
/// only read.
struct Accesses {
    written: HashSet<String>,
    loaded: HashSet<String>,
    /// Variables whose address, or a value made from it, reaches anything but the address of a load
    /// or the instructions of `ADDRESS_ARITHMETIC`: a store of any kind (`st.local` and `st.param`
    /// too, as data or as the address), an atomic, a reduction, a call, a copy, a prefetch, or any
    /// other instruction. The bytes of a variable that is not in this set can only be read by the
    /// function.
    not_only_read: HashSet<String>,
}

/// A statement as the assembler reads it: its opcode (the letters, digits, `.`, `_` and `:` after its
/// guard, if it has one: `@p` or `@!p`, with spaces or not) and its operands (all that follows the
/// opcode, which needs no space before them).
fn split_statement(statement: &str) -> (&str, &str) {
    let mut rest = statement.trim_start();
    if let Some(guard) = rest.strip_prefix('@') {
        let guard = guard.trim_start();
        let guard = guard.strip_prefix('!').unwrap_or(guard).trim_start();
        let predicate = guard
            .bytes()
            .take_while(|&b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'$' | b'%'))
            .count();
        rest = guard[predicate..].trim_start();
    }
    let opcode = rest
        .bytes()
        .take_while(|&b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b':'))
        .count();
    (&rest[..opcode], &rest[opcode..])
}

/// `signature_names` are the names in the function's signature (its parameters, and its return
/// value when it is a register): a value made from a variable's address that one of them is given
/// leaves the function, so the variable is not only read.
fn find_writes_to_globals<'a>(
    code: &'a str,
    signature_names: &[&str],
    global_names: &HashSet<&'a str>,
    safe_callees: &HashSet<&str>,
) -> Accesses {
    let items = match scan::items(code.as_bytes(), 0..code.len()) {
        Some(items) => items,
        // (What the body does with a variable is not known: none of them is only read.)
        None => {
            return Accesses {
                written: HashSet::new(),
                loaded: HashSet::new(),
                not_only_read: global_names.iter().map(|&n| n.to_string()).collect(),
            };
        }
    };

    // Collect (opcode_str, operands_str) for every statement.
    struct Stmt<'a> {
        opcode: &'a str,
        #[allow(dead_code)]
        operands: &'a str,
        /// All names (including registers) in the operands.
        names: Vec<&'a str>,
        /// The names in the first (destination) operand.
        dest_names: Vec<&'a str>,
        /// The names after the first comma (source operands, including the address of a load).
        source_names: Vec<&'a str>,
    }

    let mut stmts: Vec<Stmt<'a>> = Vec::new();

    for item in &items {
        let scan::Kind::Statement { operands, .. } = &item.kind else {
            continue;
        };
        // (The scanner ends the opcode at a space: `st.global.u32[t], %r1` and `@ %p st.global.u32
        // [t], %r1` are stores too.)
        let (op, ops) = split_statement(&code[item.range.start..operands.end]);
        // A declaration names what it declares and uses nothing, unless it has an initializer
        // (a variable of the function, `.global .u64 q = generic(t)`): then it is an instruction
        // that the rules below do not know.
        if op.starts_with('.') && !ops.contains('=') {
            continue;
        }
        let names: Vec<&'a str> = scan::names(ops).collect();
        // First operand (destination): up to the first comma outside braces/brackets.
        let first_comma = {
            let mut depth = 0usize;
            let mut pos = None;
            for (i, b) in ops.bytes().enumerate() {
                match b {
                    b'{' | b'[' | b'(' => depth += 1,
                    b'}' | b']' | b')' => depth = depth.saturating_sub(1),
                    b',' if depth == 0 => {
                        pos = Some(i);
                        break;
                    }
                    _ => {}
                }
            }
            pos
        };
        let dest_str = match first_comma {
            Some(p) => &ops[..p],
            None => ops,
        };
        let dest_names: Vec<&'a str> = scan::names(dest_str).collect();
        let source_str = match first_comma {
            Some(p) => &ops[p + 1..],
            None => "",
        };
        let source_names: Vec<&'a str> = scan::names(source_str).collect();
        stmts.push(Stmt {
            opcode: op,
            operands: ops,
            names,
            dest_names,
            source_names,
        });
    }

    // Build a map from each name (register or global) to the set of global variables it could be
    // tainted by. Initially, each global variable name maps to itself.
    let mut name_to_vars: HashMap<&'a str, HashSet<&'a str>> = HashMap::new();
    for &var in global_names {
        name_to_vars.entry(var).or_default().insert(var);
    }

    // Iterate to fixed point.
    loop {
        let mut changed = false;
        for stmt in &stmts {
            let is_load = stmt.opcode.starts_with("ld.") || stmt.opcode.starts_with("ldu.");
            // Only stores to global or shared memory can write module-level variables.
            // st.param and st.local are local; they do not escape to device memory.
            let is_memory_store =
                (stmt.opcode.starts_with("st.") && !stmt.opcode.contains(".param") && !stmt.opcode.contains(".local"))
                    || stmt.opcode.starts_with("atom.")
                    || stmt.opcode.starts_with("red.");
            let is_call = stmt.opcode == "call" || stmt.opcode == "call.uni";

            // Loads, stores, calls: do not propagate taint to their destination.
            if is_load || is_memory_store || is_call {
                continue;
            }

            // Collect all variables that any source operand is tainted by.
            let mut tainted_by: HashSet<&'a str> = HashSet::new();
            for &name in &stmt.names {
                if let Some(vars) = name_to_vars.get(name) {
                    tainted_by.extend(vars.iter().copied());
                }
            }
            if tainted_by.is_empty() {
                continue;
            }

            // Taint each destination with those variables, whatever its name: a register needs no
            // `%` (`.reg .b64 p; mov.u64 p, t;`).
            for &dest in &stmt.dest_names {
                let entry = name_to_vars.entry(dest).or_default();
                for &var in &tainted_by {
                    if entry.insert(var) {
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Determine which variables are written: a store to addressable memory (not st.param, st.local),
    // atomic, reduction, or call where any operand is tainted.
    let mut written: HashSet<String> = HashSet::new();
    for stmt in &stmts {
        let is_store =
            (stmt.opcode.starts_with("st.") && !stmt.opcode.contains(".param") && !stmt.opcode.contains(".local"))
                || stmt.opcode.starts_with("atom.")
                || stmt.opcode.starts_with("red.");
        let is_call = stmt.opcode == "call" || stmt.opcode == "call.uni";
        if !is_store && !is_call {
            continue;
        }
        if is_call {
            let target_is_safe = stmt
                .names
                .iter()
                .any(|n| !n.starts_with('%') && safe_callees.contains(*n));
            if target_is_safe {
                continue;
            }
        }
        for &name in &stmt.names {
            if let Some(vars) = name_to_vars.get(name) {
                written.extend(vars.iter().map(|&v| v.to_string()));
            }
        }
    }

    // Determine which variables are loaded from: a load where the address operand (a source
    // name after the first comma) is tainted by the variable.
    let mut loaded_from: HashSet<String> = HashSet::new();
    for stmt in &stmts {
        let is_load = stmt.opcode.starts_with("ld.") || stmt.opcode.starts_with("ldu.");
        if !is_load {
            continue;
        }
        for &name in &stmt.source_names {
            if let Some(vars) = name_to_vars.get(name) {
                loaded_from.extend(vars.iter().map(|&v| v.to_string()));
            }
        }
    }

    // Determine which variables are not only read: a tainted name anywhere but in the address of a
    // load or in an instruction that only computes a value from it.
    // (A tainted name can be in a load only in its address: the load reads. A store of the address
    // to local or parameter space is a store like any other here, so what is loaded back from there
    // is never needed to be followed.)
    let mut not_only_read: HashSet<String> = HashSet::new();
    for stmt in &stmts {
        let is_load = stmt.opcode.starts_with("ld.") || stmt.opcode.starts_with("ldu.");
        if is_load || is_address_arithmetic(stmt.opcode) {
            continue;
        }
        for &name in &stmt.names {
            if let Some(vars) = name_to_vars.get(name) {
                not_only_read.extend(vars.iter().map(|&v| v.to_string()));
            }
        }
    }
    // A register of the signature that holds a value made from the address: the function gives it
    // back to its caller (whose `call` does not taint what it receives) or keeps it in a parameter.
    for &name in signature_names {
        if let Some(vars) = name_to_vars.get(name) {
            not_only_read.extend(vars.iter().map(|&v| v.to_string()));
        }
    }

    Accesses {
        written,
        loaded: loaded_from,
        not_only_read,
    }
}

// ---------------------------------------------------------------------------
// Module-level parsing
// ---------------------------------------------------------------------------

/// Find the byte offset where the header ends (public entry point for other crates).
pub fn find_header_end_pub(text: &[u8]) -> usize {
    find_header_end(text)
}

/// Find the byte offset where the header ends: right after the last header directive (`.version`,
/// `.target`, `.address_size`) or leading comment, before the first declaration or definition.
fn find_header_end(text: &[u8]) -> usize {
    let mut last_header_end = 0;
    let mut at = 0;
    let max_lines = 64;
    for _ in 0..max_lines {
        if at >= text.len() {
            break;
        }
        let line_end = text[at..]
            .iter()
            .position(|&b| b == b'\n')
            .map_or(text.len(), |p| at + p);
        let line = &text[at..line_end];
        let indent = line.iter().take_while(|b| b.is_ascii_whitespace()).count();
        let trimmed = &line[indent..];

        if trimmed.is_empty()
            || trimmed.starts_with(b"//")
            || trimmed.starts_with(b".version")
            || trimmed.starts_with(b".target")
            || trimmed.starts_with(b".address_size")
        {
            last_header_end = if line_end < text.len() {
                line_end + 1
            } else {
                text.len()
            };
            at = last_header_end;
            continue;
        }
        break;
    }
    last_header_end
}

/// Recognise a launch-supplied shared array declared as `name[];`.
fn is_unsized_array(declaration: &str, name: &str) -> bool {
    declaration.split_once('[').is_some_and(|(prefix, suffix)| {
        !prefix.contains('=')
            && prefix.split_ascii_whitespace().any(|token| token == ".shared")
            && prefix.split_ascii_whitespace().next_back() == Some(name)
            && suffix
                .bytes()
                .filter(|b| !b.is_ascii_whitespace())
                .eq(b"];".iter().copied())
    })
}

/// Extract the variable name from a declaration like `.global .align 4 .b8 table[4] = {...};`.
fn extract_variable_name(decl: &str) -> Option<String> {
    let bytes = decl.as_bytes();
    let mut last_name = None;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' || bytes[i] == b'=' || bytes[i] == b';' {
            break;
        }
        if (bytes[i].is_ascii_alphabetic() || matches!(bytes[i], b'_' | b'$' | b'%'))
            && (i == 0 || !is_name_byte(bytes[i - 1]))
        {
            let start = i;
            while i < bytes.len() && is_name_byte(bytes[i]) {
                i += 1;
            }
            let word = &decl[start..i];
            // Skip known keywords and type specifiers.
            if !word.starts_with('.')
                && !matches!(
                    word,
                    "global"
                        | "const"
                        | "shared"
                        | "extern"
                        | "weak"
                        | "common"
                        | "visible"
                        | "align"
                        | "b8"
                        | "b16"
                        | "b32"
                        | "b64"
                        | "u8"
                        | "u16"
                        | "u32"
                        | "u64"
                        | "s8"
                        | "s16"
                        | "s32"
                        | "s64"
                        | "f16"
                        | "f16x2"
                        | "f32"
                        | "f64"
                        | "pred"
                        | "texref"
                        | "surfref"
                        | "samplerref"
                        | "v2"
                        | "v4"
                )
            {
                last_name = Some(word.to_string());
            }
            continue;
        }
        i += 1;
    }
    last_name
}

/// Scan the text between the header and the first function definition for module-level variable
/// declarations. Also scan for `.extern .func` declarations.
///
/// This operates on the region between `header_end` and the first `FuncDef` range start, plus the
/// gaps between function definitions.
/// Result of scanning module-level items between function definitions.
struct ModuleLevelItems {
    vars: Vec<VarDecl>,
    func_decls: Vec<FuncDecl>,
    file_directives: Vec<Range<usize>>,
    has_unrecognised: bool,
    has_module_pragma: bool,
    has_unindexed_shared: bool,
    /// A function definition that `crate::visit_functions` did not find: its body is not analysed.
    has_unscanned_definition: bool,
}

/// Whether a variable declaration declares more than one name (a comma outside its initializer's
/// braces, an array's brackets or parentheses).
fn declares_several(declaration: &str) -> bool {
    let mut depth = 0usize;
    declaration.bytes().any(|b| {
        match b {
            b'{' | b'[' | b'(' => depth += 1,
            b'}' | b']' | b')' => depth = depth.saturating_sub(1),
            b',' if depth == 0 => return true,
            _ => {}
        }
        false
    })
}

fn find_module_level_items(text: &[u8], header_end: usize, func_defs: &[FuncDef]) -> ModuleLevelItems {
    let mut result = ModuleLevelItems {
        vars: Vec::new(),
        func_decls: Vec::new(),
        file_directives: Vec::new(),
        has_unrecognised: false,
        has_module_pragma: false,
        has_unindexed_shared: false,
        has_unscanned_definition: false,
    };

    // Build gap ranges: regions not covered by any function definition.
    let mut gaps = Vec::new();
    let mut cursor = header_end;
    for func in func_defs {
        let range = &func.range;
        if range.start > cursor {
            gaps.push(cursor..range.start);
        }
        cursor = range.end;
    }
    if cursor < text.len() {
        gaps.push(cursor..text.len());
    }

    for gap in gaps {
        let mut at = gap.start;
        while at < gap.end {
            if let Some(next) = scan::skip_comment(text, at) {
                at = next;
                continue;
            }
            if text[at].is_ascii_whitespace() {
                at += 1;
                continue;
            }

            // `.file` directive: keep for debug info.
            let rest = &text[at..gap.end];
            if rest.starts_with(b".file") && rest.get(5).is_some_and(u8::is_ascii_whitespace) {
                let start = at;
                while at < gap.end && text[at] != b'\n' {
                    at += 1;
                }
                if at < gap.end {
                    at += 1;
                }
                result.file_directives.push(start..at);
                continue;
            }

            // nvcc's .loc records may name strings defined in this DWARF section.
            // Keep the labels and their bytes together with the .file directives.
            if let Some(after) = rest.strip_prefix(b".section")
                && after.first().is_some_and(u8::is_ascii_whitespace)
                && let Some(tail) = after.trim_ascii_start().strip_prefix(b".debug_str")
                && tail.trim_ascii_start().starts_with(b"{")
            {
                let open = gap.end - tail.trim_ascii_start().len();
                if let Some(close) = crate::matching(text, open).filter(|&i| i < gap.end) {
                    let end = close + 1 + usize::from(text.get(close + 1) == Some(&b'\n'));
                    result.file_directives.push(at..end);
                    at = end;
                    continue;
                }
            }

            // Comments that were not caught by skip_comment (line start with `//`).
            if text[at..gap.end].starts_with(b"//") {
                while at < gap.end && text[at] != b'\n' {
                    at += 1;
                }
                if at < gap.end {
                    at += 1;
                }
                continue;
            }

            // Detect linkage prefix: `.visible`, `.weak`, `.extern`.
            let rest = &text[at..gap.end];
            let (has_visible, has_weak, has_extern, after_prefix) = parse_linkage(rest);

            // Check if this is a variable declaration: linkage? + (.global | .const | .shared | .common).
            let after = &rest[after_prefix..];
            let is_var_keyword = after.starts_with(b".global")
                || after.starts_with(b".const")
                || after.starts_with(b".shared")
                || after.starts_with(b".common");

            // `.extern .func` or `.visible .func` or `.weak .func` or plain `.func` (declaration, no body).
            let is_func_keyword = after.starts_with(b".func") || after.starts_with(b".entry");
            let is_func_decl_possible = is_func_keyword && (has_extern || has_visible || has_weak || after_prefix == 0);

            if is_var_keyword {
                let start = at;
                let mut depth = 0;
                while at < gap.end {
                    if text[at] == b'{' {
                        depth += 1;
                    } else if text[at] == b'}' {
                        depth -= 1;
                    } else if text[at] == b';' && depth == 0 {
                        at += 1;
                        break;
                    }
                    at += 1;
                }
                if at < gap.end && text[at] == b'\n' {
                    at += 1;
                }
                let decl_text = std::str::from_utf8(&text[start..at]).unwrap_or("");
                let shared = after.starts_with(b".shared");
                // The scanner records one name per declaration. Never infer shared-memory
                // bindings from a variable list or a declaration it could not finish.
                if shared && (decl_text.contains(',') || !decl_text.trim_end().ends_with(';')) {
                    result.has_unindexed_shared = true;
                }
                if let Some(name) = extract_variable_name(decl_text) {
                    // An unsized extern shared array denotes the storage supplied at launch.
                    // Externs with fixed extents and definitions in other inputs need the linker.
                    let linkage =
                        if shared && has_extern && !has_visible && !has_weak && is_unsized_array(decl_text, &name) {
                            Linkage::DynamicShared
                        } else if has_visible || has_extern || after.starts_with(b".common") {
                            Linkage::External
                        } else if has_weak {
                            Linkage::Weak
                        } else {
                            Linkage::Private
                        };
                    // (One name per declaration: `.global .b8 t[4] = {...}, u[4] = {...};` declares
                    // a `u` that nothing here knows of, and that a copy of `t` would hold.)
                    let initialized_data = linkage == Linkage::Private
                        && !shared
                        && decl_text.contains('=')
                        && !declares_several(decl_text)
                        && !decl_text.contains(".attribute")
                        && ![".texref", ".samplerref", ".surfref"]
                            .iter()
                            .any(|kind| decl_text.contains(kind));
                    result.vars.push(VarDecl {
                        range: start..at,
                        name,
                        linkage,
                        persistent: !shared,
                        initialized_data,
                    });
                } else if shared {
                    result.has_unindexed_shared = true;
                }
                continue;
            }

            if is_func_decl_possible {
                // A function declaration ended by `;` (no body) at module level.
                let start = at;
                while at < gap.end && text[at] != b';' && text[at] != b'{' {
                    at += 1;
                }
                if at < gap.end && text[at] == b';' {
                    at += 1; // past `;`
                    if at < gap.end && text[at] == b'\n' {
                        at += 1;
                    }
                    let decl = std::str::from_utf8(&text[start..at]).unwrap_or("");
                    if let Some(name) = extract_func_decl_name(decl) {
                        result.func_decls.push(FuncDecl { range: start..at, name });
                    }
                } else {
                    // Has a body: this is a definition, not a declaration. Should have been found
                    // by crate::functions() (it is not when no space follows the keyword:
                    // `.func(.reg .b32 r) f()`). Skip past the body, which is not analysed.
                    result.has_unscanned_definition = true;
                    if at < gap.end && text[at] == b'{' {
                        at = crate::matching(text, at).map_or(gap.end, |c| c + 1);
                        if at < gap.end && text[at] == b'\n' {
                            at += 1;
                        }
                    }
                }
                continue;
            }

            // Known directives/items that can appear at module level (or leak into gaps from function
            // signatures when the gap covers the region between the .entry keyword and {) and are safe.
            if text[at..gap.end].starts_with(b".section")
                || text[at..gap.end].starts_with(b".loc ")
                || (rest.starts_with(b".pragma") && rest.get(7).is_some_and(u8::is_ascii_whitespace))
                || text[at..gap.end].starts_with(b".param ")
                || text[at..gap.end].starts_with(b".reg ")
                || text[at..gap.end].starts_with(b".maxntid")
                || text[at..gap.end].starts_with(b".minnctapersm")
                || text[at..gap.end].starts_with(b".reqntid")
                || text[at..gap.end].starts_with(b".maxnreg")
                || text[at] == b')'
                || text[at] == b'{'
                || text[at] == b'}'
                || text[at] == b';'
            {
                while at < gap.end && text[at] != b'\n' {
                    at += 1;
                }
                if at < gap.end {
                    at += 1;
                }
                continue;
            }

            // Anything else in a gap: module-level text we do not parse. Gaps may contain pieces
            // of function signatures that the gap calculation does not fully cover (parameter
            // lists, function attributes, instructions). Only flag as unrecognised if the line
            // starts with a dot that does not match any known directive -- a genuine unknown
            // directive. Everything else (instructions, braces, parameters) is part of the
            // surrounding code and is safe.
            let start = at;
            while at < gap.end && text[at] != b'\n' {
                at += 1;
            }
            if at < gap.end {
                at += 1;
            }
            let line = std::str::from_utf8(&text[start..at]).unwrap_or("").trim();
            if line.starts_with('.') && !line.starts_with("//") {
                // A dot-directive we do not recognise. Not safe to split.
                result.has_unrecognised = true;
            }
        }
    }

    // PTX directives need not start on a new line. The header and debug parsers consume
    // whole lines, so check for link-relevant tokens they might have skipped (e.g. a
    // .shared definition after .address_size or .file). Function bodies stay unexamined.
    let mut covered: Vec<_> = func_defs
        .iter()
        .map(|f| &f.range)
        .chain(result.vars.iter().map(|v| &v.range))
        .chain(result.func_decls.iter().map(|f| &f.range))
        .collect();
    covered.sort_unstable_by_key(|r| r.start);
    let mut start = 0;
    for range in covered.into_iter().chain(std::iter::once(&(text.len()..text.len()))) {
        let mut at = start;
        while at < range.start {
            if text[at] == b'"' {
                at = scan::skip_string(text, at);
            } else if is_name_byte(text[at]) {
                let token = at;
                at += 1;
                while at < range.start && is_name_byte(text[at]) {
                    at += 1;
                }
                match &text[token..at] {
                    b".pragma" => result.has_module_pragma = true,
                    b".shared" => result.has_unindexed_shared = true,
                    _ => {}
                }
            } else {
                at += 1;
            }
        }
        start = start.max(range.end);
    }
    result
}

/// Parse optional linkage prefix at the start of `text`. Returns (has_visible, has_weak,
/// has_extern, bytes_consumed).
fn parse_linkage(text: &[u8]) -> (bool, bool, bool, usize) {
    let mut vis = false;
    let mut weak = false;
    let mut ext = false;
    let mut at = 0;
    loop {
        let rest = &text[at..];
        if rest.starts_with(b".visible") && rest.get(8).is_some_and(|b| b.is_ascii_whitespace()) {
            vis = true;
            at += 8;
        } else if rest.starts_with(b".weak") && rest.get(5).is_some_and(|b| b.is_ascii_whitespace()) {
            weak = true;
            at += 5;
        } else if rest.starts_with(b".extern") && rest.get(7).is_some_and(|b| b.is_ascii_whitespace()) {
            ext = true;
            at += 7;
        } else {
            break;
        }
        // Skip whitespace after the keyword.
        while at < text.len() && text[at].is_ascii_whitespace() {
            at += 1;
        }
    }
    (vis, weak, ext, at)
}

/// Extract the name from a function declaration. Handles `.extern .func (rettype) name (params);`
/// and variants with `.visible`, `.weak`. The name is the first identifier outside parens that
/// is not a keyword.
fn extract_func_decl_name(decl: &str) -> Option<String> {
    let bytes = decl.as_bytes();
    let mut i = 0;
    let mut depth: usize = 0;
    let mut found_keyword = false;
    while i < bytes.len() {
        if bytes[i] == b'(' {
            depth += 1;
            i += 1;
            continue;
        }
        if bytes[i] == b')' {
            depth = depth.saturating_sub(1);
            i += 1;
            continue;
        }
        if bytes[i] == b';' {
            break;
        }
        let can_start = bytes[i].is_ascii_alphabetic() || matches!(bytes[i], b'.' | b'_' | b'$' | b'%');
        if depth == 0 && can_start && (i == 0 || !is_name_byte(bytes[i - 1])) {
            let start = i;
            while i < bytes.len() && is_name_byte(bytes[i]) {
                i += 1;
            }
            let word = &decl[start..i];
            if matches!(
                word,
                ".extern"
                    | ".func"
                    | ".visible"
                    | ".weak"
                    | ".entry"
                    | ".param"
                    | "extern"
                    | "func"
                    | "visible"
                    | "weak"
            ) {
                found_keyword = true;
                continue;
            }
            if word.starts_with('.') {
                continue;
            }
            if found_keyword {
                return Some(word.to_string());
            }
            continue;
        }
        i += 1;
    }
    None
}

/// Check if an entry function (transitively) references a variable. Static version for use during
/// construction.
fn entry_references_var_static(
    entry: &str,
    var: &str,
    func_refs: &HashMap<String, HashSet<String>>,
    var_refs: &HashMap<String, HashSet<String>>,
) -> bool {
    let mut visited = HashSet::new();
    let mut stack = vec![entry.to_string()];
    while let Some(func) = stack.pop() {
        if !visited.insert(func.clone()) {
            continue;
        }
        if let Some(vr) = var_refs.get(&func)
            && vr.contains(var)
        {
            return true;
        }
        if let Some(fr) = func_refs.get(&func) {
            for callee in fr {
                stack.push(callee.clone());
            }
        }
    }
    false
}

/// Locate definitions and module-level declarations without analysing any instructions.
fn module_structure(text_str: &str) -> (usize, Vec<FuncDef>, Vec<String>, ModuleLevelItems) {
    let text = text_str.as_bytes();
    let header_end = find_header_end(text);

    let mut func_defs = Vec::new();
    let mut entries = Vec::new();

    crate::visit_functions(text, |directive, func| {
        let name = text_str[func.name.clone()].to_string();
        let body = func.body.clone();

        let def_start = find_def_start(text, directive);
        let def_end = func.body.end;
        let def_end = if def_end < text.len() && text[def_end] == b'\n' {
            def_end + 1
        } else {
            def_end
        };

        let is_entry = func.entry;

        func_defs.push(FuncDef {
            range: def_start..def_end,
            name: name.clone(),
            signature: directive..body.start,
            body,
            is_entry,
        });

        if is_entry {
            entries.push(name);
        }
    });

    let module_items = find_module_level_items(text, header_end, &func_defs);
    (header_end, func_defs, entries, module_items)
}

/// Link-visible variables across all the objects of a device link. Identical weak shared
/// definitions and matching declarations of launch-supplied shared memory can be isolated.
#[derive(Default)]
pub struct LinkedShared {
    definitions: HashMap<String, Option<Vec<u8>>>,
}

impl LinkedShared {
    /// Return this object's entry names while recording the declarations that could bind
    /// another object's shared symbol. Leave unrecognised module structure to the linker.
    pub fn add_object(&mut self, source: &[u8]) -> Option<Vec<String>> {
        let source_text = std::str::from_utf8(source).ok()?;
        let blanked = source_text.contains('/').then(|| scan::blank_comments(source_text));
        let (_, _, entries, items) = module_structure(blanked.as_deref().unwrap_or(source_text));
        if items.has_unrecognised || items.has_unindexed_shared {
            return None;
        }
        for var in items.vars {
            if var.linkage == Linkage::Private {
                continue;
            }
            let binding = (!var.persistent && matches!(var.linkage, Linkage::Weak | Linkage::DynamicShared))
                .then(|| source[var.range].trim_ascii());
            self.definitions
                .entry(var.name)
                .and_modify(|existing| {
                    if existing.as_deref() != binding {
                        *existing = None;
                    }
                })
                .or_insert_with(|| binding.map(<[u8]>::to_vec));
        }
        Some(entries)
    }

    fn allows(&self, var: &VarDecl, source: &[u8]) -> bool {
        !var.persistent
            && match var.linkage {
                Linkage::Private => true,
                Linkage::Weak | Linkage::DynamicShared => {
                    self.definitions.get(&var.name).and_then(Option::as_deref)
                        == Some(source[var.range.clone()].trim_ascii())
                }
                Linkage::External => false,
            }
    }
}

fn plain_identifier(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes
        .next()
        .is_some_and(|b| b.is_ascii_alphabetic() || matches!(b, b'_' | b'$'))
        && bytes.all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'$'))
}

/// A complete PTX debug location line. An unrecognised extension or trailing text is
/// kept intact, even if it starts with the same directive.
pub(crate) fn debug_location(line: &str) -> bool {
    let mut clauses = line.split(',');
    let position = |clause: &str, directive: &str| {
        let mut fields = clause.split_ascii_whitespace();
        fields.next() == Some(directive)
            && (0..3).all(|_| fields.next().is_some_and(|s| s.bytes().all(|b| b.is_ascii_digit())))
            && fields.next().is_none()
    };
    if !position(clauses.next().unwrap_or_default(), ".loc") {
        return false;
    }
    let Some(function) = clauses.next() else {
        return true;
    };
    let mut fields = function.split_ascii_whitespace();
    fields.next() == Some("function_name")
        && fields.next().is_some_and(plain_identifier)
        && fields.next().is_none()
        && clauses.next().is_none_or(|inline| position(inline, "inlined_at"))
        && clauses.next().is_none()
}

impl ModuleInfo {
    /// Analyse a PTX module. The text should be the full module content (already lowered for Volta
    /// if needed).
    pub fn analyse(source: &[u8]) -> ModuleInfo {
        // The declaration scanner, statement scanner and operand analysis must see the same
        // tokens. Keep byte offsets and newlines so extraction can copy the original source.
        let source_text = std::str::from_utf8(source).unwrap_or("");
        let blanked = source_text.contains('/').then(|| scan::blank_comments(source_text));
        let text_str = blanked.as_deref().unwrap_or(source_text);
        let (header_end, func_defs, entries, module_items) = module_structure(text_str);
        let variables = module_items.vars;
        let func_decls = module_items.func_decls;
        let file_directives = module_items.file_directives;
        let has_unrecognised_items = module_items.has_unrecognised;

        // Collect all known module-level names for reference checking. Only persistent variables
        // (.global, .const) need write analysis; .shared is per-launch and safe to duplicate.
        let all_var_names: HashSet<&str> = variables.iter().map(|v| v.name.as_str()).collect();
        let persistent_names: HashSet<String> = variables
            .iter()
            .filter(|v| v.persistent)
            .map(|v| v.name.clone())
            .collect();
        let persistent_names_ref: HashSet<&str> = persistent_names.iter().map(String::as_str).collect();
        let all_func_names: HashSet<&str> = func_defs
            .iter()
            .map(|f| f.name.as_str())
            .chain(func_decls.iter().map(|d| d.name.as_str()))
            .collect();

        let defined_funcs: HashSet<&str> = func_defs.iter().map(|f| f.name.as_str()).collect();
        let safe_callees: HashSet<&str> = func_decls
            .iter()
            .filter(|d| READONLY_EXTERN_CALLEES.contains(&d.name.as_str()))
            .filter(|d| !defined_funcs.contains(d.name.as_str()))
            .map(|d| d.name.as_str())
            .collect();

        let mut unbounded_calls = HashSet::new();
        let mut targets = Vec::new();
        if text_str.contains(".callprototype") {
            for f in func_defs.iter().filter(|f| !f.is_entry) {
                let signature = text_str[f.signature.clone()]
                    .strip_prefix(".func")
                    .and_then(|text| calls::Signature::parse(text, &f.name));
                targets.push((&f.name, signature));
            }
        }
        let mut func_refs_map: HashMap<String, HashSet<String>> = HashMap::new();
        let mut var_refs_map: HashMap<String, HashSet<String>> = HashMap::new();
        let mut written_vars = HashSet::new();
        let mut loaded_vars = HashSet::new();
        let mut not_only_read: HashSet<String> = HashSet::new();
        // A body that is not analysed could do anything with any variable.
        if module_items.has_unscanned_definition {
            not_only_read.extend(persistent_names.iter().cloned());
        }

        // A referenced variable brings every symbol used to initialize it. A persistent
        // address in an initializer can also be loaded and written through.
        let declared = |name: &str| all_var_names.contains(name) || all_func_names.contains(name);
        let mut initializer_refs: HashMap<&str, HashSet<&str>> = HashMap::new();
        let mut escaped_vars = HashSet::new();
        for var in &variables {
            let Some((_, initializer)) = text_str[var.range.clone()].split_once('=') else {
                continue;
            };
            let refs = find_referenced_names(initializer, declared);
            for &name in &refs {
                if persistent_names_ref.contains(name) {
                    escaped_vars.insert(name.to_string());
                }
            }
            if !refs.is_empty() {
                initializer_refs.entry(var.name.as_str()).or_default().extend(refs);
            }
        }
        written_vars.extend(escaped_vars.iter().cloned());

        // The write/load taint analysis is only needed when the module has persistent variables.
        // Modules without persistent variables have no coherence concern, and the taint analysis
        // over an empty set of globals is a no-op anyway.
        let need_taint = !persistent_names_ref.is_empty();

        struct FuncResult {
            name: String,
            fn_refs: HashSet<String>,
            var_refs: HashSet<String>,
            unbounded: bool,
            accesses: Option<Accesses>,
            unscanned: bool,
        }

        let analyse_func = |func: &FuncDef| -> FuncResult {
            let body_text = &text_str[func.body.clone()];
            let mut all_names = find_referenced_names(body_text, declared);
            if !initializer_refs.is_empty() {
                let mut pending: Vec<_> = all_names.iter().filter_map(|&n| initializer_refs.get(n)).collect();
                while let Some(refs) = pending.pop() {
                    for &name in refs {
                        if all_names.insert(name)
                            && let Some(next) = initializer_refs.get(name)
                        {
                            pending.push(next);
                        }
                    }
                }
            }
            let mut fn_refs: HashSet<String> = all_names
                .iter()
                .filter(|n| all_func_names.contains(**n))
                .map(|&n| n.to_string())
                .collect();
            let mut unbounded = false;
            if body_text.contains(".callprototype") {
                let possible = scan::items(text_str.as_bytes(), func.body.clone())
                    .map(|items| calls::used(text_str, &items))
                    .unwrap_or_else(|| HashSet::from([None]));
                if !possible.is_empty() {
                    // Keep runtime pointers in one loaded module. The ordinary call graph
                    // then propagates through each compatible target and its own callees.
                    unbounded = true;
                    for (name, target) in &targets {
                        if possible
                            .iter()
                            .any(|proto| proto.as_ref().zip(target.as_ref()).is_none_or(|(a, b)| a.compatible(b)))
                        {
                            fn_refs.insert((*name).clone());
                        }
                    }
                }
            }
            let vr: HashSet<String> = all_names
                .iter()
                .filter(|n| all_var_names.contains(**n))
                .map(|&n| n.to_string())
                .collect();
            let mut accesses = None;
            let mut unscanned = false;
            if need_taint {
                let signature: Vec<&str> = scan::names(&text_str[func.signature.clone()]).collect();
                let touches_persistent = vr.iter().any(|n| persistent_names.contains(n))
                    || signature.iter().any(|&n| persistent_names_ref.contains(n));
                if touches_persistent {
                    accesses = Some(find_writes_to_globals(
                        body_text,
                        &signature,
                        &persistent_names_ref,
                        &safe_callees,
                    ));
                } else if scan::items(body_text.as_bytes(), 0..body_text.len()).is_none() {
                    unscanned = true;
                }
            }
            FuncResult {
                name: func.name.clone(),
                fn_refs,
                var_refs: vr,
                unbounded,
                accesses,
                unscanned,
            }
        };

        for func in &func_defs {
            let res = analyse_func(func);
            if res.unbounded {
                unbounded_calls.insert(res.name.clone());
            }
            if let Some(accesses) = res.accesses {
                written_vars.extend(accesses.written);
                loaded_vars.extend(accesses.loaded);
                not_only_read.extend(accesses.not_only_read);
            }
            if res.unscanned {
                not_only_read.extend(persistent_names.iter().cloned());
            }
            func_refs_map.insert(res.name.clone(), res.fn_refs);
            var_refs_map.insert(res.name, res.var_refs);
        }

        // The variables of which every copy keeps the bytes the module gives it: plain data with an
        // initializer that names nothing, whose address no function stores, passes, copies or gives
        // to anything but a load, and no other variable holds.
        let read_only_vars: HashSet<String> = variables
            .iter()
            .filter(|v| v.persistent && v.initialized_data)
            .filter(|v| !initializer_refs.contains_key(v.name.as_str()))
            // (`written_vars` holds `escaped_vars`.)
            .filter(|v| !written_vars.contains(&v.name) && !not_only_read.contains(&v.name))
            .map(|v| v.name.clone())
            .collect();

        ModuleInfo {
            text: source.to_vec(),
            header: 0..header_end,
            file_directives,
            variables,
            func_decls,
            func_defs,
            entries,
            func_refs: func_refs_map,
            unbounded_calls,
            var_refs: var_refs_map,
            written_vars,
            escaped_vars,
            loaded_vars,
            read_only_vars,
            persistent_names,
            has_unrecognised_items,
            has_module_pragma: module_items.has_module_pragma,
        }
    }

    /// Whether this module has any independent entries that can be compiled on demand. With
    /// per-entry classification, a module is "splittable" if at least one entry is independent.
    /// Dependent entries come from the whole module (materialized once).
    pub fn splittable(&self) -> Splittable {
        if self.has_module_pragma {
            return Splittable::No {
                reason: "module-level pragma".into(),
            };
        }
        if self.entries.len() <= 1 {
            return Splittable::Yes;
        }
        if self.has_unrecognised_items {
            return Splittable::No {
                reason: "unrecognised module-level items".to_string(),
            };
        }
        // All persistent variables are host-reachable (measured on V100 driver 580:
        // cuModuleGetGlobal finds every .global and .const, not just .visible ones). Per-entry
        // classification handles this: entries that reference a persistent variable are dependent
        // and come from the materialized whole module. This method checks only module-level
        // conditions (escaped vars, shared mutable state); the per-entry gate is
        // `is_entry_independent`.
        if !self.escaped_vars.is_empty() {
            let mut escaped: Vec<&str> = self.escaped_vars.iter().map(|s| s.as_str()).collect();
            escaped.sort();
            return Splittable::No {
                reason: format!("variables with address in an initializer (escaped): {escaped:?}"),
            };
        }
        let mut shared_mutable = Vec::new();
        for var in &self.written_vars {
            let mut users = 0;
            for entry in &self.entries {
                if self.entry_references_var(entry, var) {
                    users += 1;
                }
            }
            if users > 1 {
                shared_mutable.push(var.clone());
            }
        }
        if shared_mutable.is_empty() {
            Splittable::Yes
        } else {
            shared_mutable.sort();
            Splittable::No {
                reason: format!("shared mutable variables: {shared_mutable:?}"),
            }
        }
    }

    /// Check if an entry function (transitively through device functions) references a variable.
    fn entry_references_var(&self, entry: &str, var: &str) -> bool {
        entry_references_var_static(entry, var, &self.func_refs, &self.var_refs)
    }

    /// The transitive closure of device functions referenced by `entry`.
    fn transitive_callees(&self, entry: &str) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut stack = vec![entry.to_string()];
        while let Some(func) = stack.pop() {
            if !visited.insert(func.clone()) {
                continue;
            }
            if let Some(fr) = self.func_refs.get(&func) {
                for callee in fr {
                    stack.push(callee.clone());
                }
            }
        }
        visited.remove(entry);
        visited
    }

    fn may_call_unknown(&self, entry: &str, callees: &HashSet<String>) -> bool {
        self.unbounded_calls.contains(entry) || !self.unbounded_calls.is_disjoint(callees)
    }

    /// The transitive closure of module-level variables referenced by `entry` and its callees.
    fn transitive_vars(&self, entry: &str, callees: &HashSet<String>) -> HashSet<String> {
        let mut vars = HashSet::new();
        for func in std::iter::once(entry).chain(callees.iter().map(String::as_str)) {
            if let Some(vr) = self.var_refs.get(func) {
                vars.extend(vr.iter().cloned());
            }
        }
        vars
    }

    /// Whether an entry has statically known callees and references no persistent variable (.global, .const)
    /// transitively, except for message strings whose address flows only into vprintf/__assertfail
    /// and read-only data (`read_only_vars`).
    ///
    /// An independent entry compiles on demand in its own sub-module (with any message strings and
    /// read-only data duplicated: `copied_tables` names the latter, which the program must not be
    /// given by name once a sub-module holds a copy). A dependent entry comes from the whole module
    /// (materialized once).
    pub fn is_entry_independent(&self, entry: &str) -> bool {
        let callees = self.transitive_callees(entry);
        if self.has_module_pragma || self.may_call_unknown(entry, &callees) {
            return false;
        }
        let vars = self.transitive_vars(entry, &callees);
        let persistent_refs: Vec<&String> = vars.iter().filter(|v| self.persistent_names.contains(*v)).collect();
        if persistent_refs.is_empty() {
            return true;
        }
        // Check if all persistent references are message strings (variables whose address flows
        // only into safe callees, vprintf and __assertfail, and is never the address of a load or
        // store in kernel code) or read-only data.
        persistent_refs
            .iter()
            .all(|v| self.is_message_string_only(v) || self.read_only_vars.contains(*v))
    }

    /// The read-only data (`read_only_vars`) that the sub-module of `entry` holds a copy of, in the
    /// order of their names: the variables that kernels compiled alone and the whole module each
    /// have a copy of, which stay the same only while the program writes none.
    pub fn copied_tables(&self, entry: &str) -> Vec<String> {
        let mut tables: Vec<String> = self
            .transitive_vars(entry, &self.transitive_callees(entry))
            .into_iter()
            .filter(|v| self.read_only_vars.contains(v) && !self.is_message_string_only(v))
            .collect();
        tables.sort();
        tables
    }

    /// Extract a leaf kernel from a relocatable object. A call (including an indirect call),
    /// an address of a device function, or a reference to persistent storage can depend on the
    /// complete linked image. Leave all of those to the device linker.
    pub fn extract_linked_leaf(&self, name: &str, shared: &LinkedShared) -> Option<Vec<u8>> {
        self.linked_leaf(name, shared)?;
        self.extract(name)
    }

    /// Comparison key for a weak leaf, including its header, signature and referenced
    /// shared storage. Local branch labels and source locations may differ between objects.
    /// The selected definition and its debug tables are emitted unchanged by `extract`.
    pub fn weak_leaf_definition(&self, name: &str, shared: &LinkedShared) -> Option<Vec<u8>> {
        let entry = self.linked_leaf(name, shared)?;
        let (_, weak, _, _) = parse_linkage(&self.text[entry.range.start..entry.body.start]);
        if !weak {
            return None;
        }
        let key = self.weak_body_key(entry);
        self.extract_with_debug(name, &[], key.as_deref().map(|bytes| (entry, bytes)))
    }

    /// A key for direct branches to unique outer-scope labels. Every occurrence of a
    /// candidate name must be either its definition or the sole operand of a direct branch.
    /// Leave an ambiguous scope, a symbol in another role, or an unknown form to byte equality.
    fn weak_body_key(&self, entry: &FuncDef) -> Option<Vec<u8>> {
        let raw = std::str::from_utf8(&self.text[entry.body.clone()]).ok()?;
        let blanked = raw.contains('/').then(|| scan::blank_comments(raw));
        let body = blanked.as_deref().unwrap_or(raw);
        let items = scan::items(body.as_bytes(), 0..body.len())?;
        let mut labels = HashMap::new();
        let mut depth = 0usize;
        for item in &items {
            match item.kind {
                scan::Kind::Open => depth += 1,
                scan::Kind::Close => depth = depth.checked_sub(1)?,
                scan::Kind::Label => {
                    let name = body[item.range.clone()].strip_suffix(':')?.trim_ascii();
                    let index = u32::try_from(labels.len()).ok()?;
                    if depth != 0 || !plain_identifier(name) || labels.insert(name, index).is_some() {
                        return None;
                    }
                }
                _ => {}
            }
        }
        let signature = std::str::from_utf8(&self.text[entry.range.start..entry.body.start]).ok()?;
        if scan::names(signature).any(|s| labels.contains_key(s))
            || self.variables.iter().any(|v| labels.contains_key(v.name.as_str()))
            || self.func_decls.iter().any(|f| labels.contains_key(f.name.as_str()))
            || self.func_defs.iter().any(|f| labels.contains_key(f.name.as_str()))
        {
            return None;
        }

        // PTX source is UTF-8. These two bytes cannot occur in any literal, identifier or
        // instruction. They separate complete items and encode fixed-width label ordinals.
        let mut key = Vec::with_capacity(raw.len());
        let mut targeted = HashSet::new();
        key.push(0xfe);
        for item in items {
            let text = &body[item.range.clone()];
            let target = match item.kind {
                scan::Kind::Line if debug_location(text) => continue,
                scan::Kind::Label => {
                    let name = text.strip_suffix(':')?.trim_ascii();
                    Some((item.range.start..item.range.start + name.len(), labels[name]))
                }
                scan::Kind::Statement { opcode, operands, .. } => match &body[opcode] {
                    "bra" | "bra.uni" => {
                        let all = &body[operands.clone()];
                        let name = all.trim_ascii();
                        let &index = labels.get(name)?;
                        let start = operands.end - all.trim_ascii_start().len();
                        if scan::names(&body[item.range.start..start]).any(|s| labels.contains_key(s)) {
                            return None;
                        }
                        targeted.insert(index);
                        Some((start..start + name.len(), index))
                    }
                    // These attach non-instruction meanings to names that look like labels.
                    ".callprototype" | ".calltargets" | ".branchtargets" => return None,
                    _ => None,
                },
                _ => None,
            };
            if let Some((range, index)) = target {
                key.extend_from_slice(&body.as_bytes()[item.range.start..range.start]);
                key.push(0xff);
                key.extend_from_slice(&index.to_le_bytes());
                key.extend_from_slice(&body.as_bytes()[range.end..item.range.end]);
            } else {
                if scan::names(text).any(|s| labels.contains_key(s)) {
                    return None;
                }
                key.extend_from_slice(text.as_bytes());
            }
            key.push(0xfe);
        }
        (targeted.len() == labels.len()).then_some(key)
    }

    fn linked_leaf(&self, name: &str, shared: &LinkedShared) -> Option<&FuncDef> {
        let entry = self.func_defs.iter().find(|f| f.is_entry && f.name == name)?;
        if self.has_unrecognised_items
            || self.has_module_pragma
            || self.func_refs.get(name).is_some_and(|refs| !refs.is_empty())
            || self.var_refs.get(name).is_some_and(|refs| {
                self.variables
                    .iter()
                    .any(|v| refs.contains(&v.name) && !shared.allows(v, &self.text))
            })
            || scan::items(&self.text, entry.body.clone())?.iter().any(|item| {
                matches!(&item.kind, scan::Kind::Statement { opcode, .. }
                    if self.text[opcode.clone()].split(|&b| b == b'.').next() == Some(b"call"))
            })
        {
            return None;
        }
        Some(entry)
    }

    /// A persistent variable qualifies as a message string if its address only flows into the
    /// arguments of vprintf/__assertfail (declared `.extern`) and is never the address operand of
    /// a load, a store, an atomic or a reduction, never stored as data, and never passed to
    /// another call.
    ///
    /// The `written_vars` set covers stores, atomics, reductions, data-as-stored, and calls to
    /// non-safe callees. The `loaded_vars` set covers loads through the variable's address. The
    /// `escaped_vars` set covers addresses that leak through another variable's initializer.
    fn is_message_string_only(&self, var: &str) -> bool {
        if self.written_vars.contains(var) {
            return false;
        }
        if self.escaped_vars.contains(var) {
            return false;
        }
        // The variable's address must never appear in the address operand of a load: a load
        // means the kernel reads the variable's data, so the variable is not a mere message
        // string whose content is irrelevant.
        if self.loaded_vars.contains(var) {
            return false;
        }
        true
    }

    /// The persistent variables that make `entry` dependent (not a message string).
    /// Empty if the entry is independent.
    pub fn dependent_vars(&self, entry: &str) -> Vec<String> {
        let vars = self.transitive_vars(entry, &self.transitive_callees(entry));
        vars.iter()
            .filter(|v| self.persistent_names.contains(*v))
            .filter(|v| !self.is_message_string_only(v) && !self.read_only_vars.contains(*v))
            .cloned()
            .collect()
    }

    /// The number of independent entries (those that can be compiled on demand).
    pub fn independent_count(&self) -> usize {
        self.entries.iter().filter(|e| self.is_entry_independent(e)).count()
    }

    /// The number of dependent entries (those that need the whole module).
    pub fn dependent_count(&self) -> usize {
        self.entries.len() - self.independent_count()
    }

    /// Extract a sub-module containing only the named entry function and its dependencies.
    /// Returns `None` if the entry is missing or needs a module pragma.
    pub fn extract(&self, entry_name: &str) -> Option<Vec<u8>> {
        self.extract_with_debug(entry_name, &self.file_directives, None)
    }

    fn extract_with_debug(
        &self,
        entry_name: &str,
        debug_directives: &[Range<usize>],
        comparison_body: Option<(&FuncDef, &[u8])>,
    ) -> Option<Vec<u8>> {
        if self.has_module_pragma || !self.entries.contains(&entry_name.to_string()) {
            return None;
        }

        let needed_funcs = self.transitive_callees(entry_name);
        let needed_vars = self.transitive_vars(entry_name, &needed_funcs);

        // A declaration or definition used as an initializer must precede that initializer.
        // Keep the original relative order of all selected items.
        let mut ranges: Vec<_> = debug_directives.iter().collect();
        for var in &self.variables {
            if needed_vars.contains(&var.name) {
                ranges.push(&var.range);
            }
        }
        for decl in &self.func_decls {
            if needed_funcs.contains(&decl.name) || decl.name == entry_name {
                ranges.push(&decl.range);
            }
        }
        for def in &self.func_defs {
            if def.name == entry_name || needed_funcs.contains(&def.name) {
                ranges.push(&def.range);
            }
        }
        ranges.sort_unstable_by_key(|range| range.start);
        let mut out = Vec::with_capacity(self.header.len() + ranges.iter().map(|r| r.len()).sum::<usize>());
        out.extend_from_slice(&self.text[self.header.clone()]);
        for range in ranges {
            if let Some((entry, key)) = comparison_body
                && range == &entry.range
            {
                out.extend_from_slice(&self.text[range.start..entry.body.start]);
                out.extend_from_slice(key);
                out.extend_from_slice(&self.text[entry.body.end..range.end]);
            } else {
                out.extend_from_slice(&self.text[range.clone()]);
            }
        }

        Some(out)
    }

    /// Names of the entry functions in this module.
    pub fn entry_names(&self) -> &[String] {
        &self.entries
    }

    /// Number of entry functions.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Each kernel with the functions it reaches: itself first, then those it references, directly or
    /// through others, in the order of their names. A call through a runtime pointer with only
    /// a prototype can reach any ABI-compatible device-function definition in the module.
    pub fn reach(&self) -> impl Iterator<Item = (&str, Vec<String>)> + '_ {
        self.entries.iter().map(|entry| {
            let mut callees: Vec<_> = self.transitive_callees(entry).into_iter().collect();
            callees.sort();
            let mut reached = Vec::with_capacity(callees.len() + 1);
            reached.push(entry.clone());
            reached.extend(callees);
            (entry.as_str(), reached)
        })
    }
}

/// References spanning CUDA relocatable objects. Private symbols are scoped to their object;
/// extern, visible, weak and common symbols meet at the same node across the link.
#[derive(Default)]
pub struct LinkedDependencies {
    objects: usize,
    nodes: HashMap<(Option<usize>, String), usize>,
    callers: Vec<Vec<usize>>,
    kernels: Vec<(usize, String)>,
    trapped: Vec<(usize, crate::Trapped)>,
    indirect: Vec<(usize, String)>,
}

impl LinkedDependencies {
    /// Add one PTX unit. Unrecognised module directives can introduce names or edges (for
    /// example an alias), so the caller must fall back when this returns false.
    #[must_use]
    pub fn add(&mut self, text: &[u8], traps: &[crate::Trapped]) -> bool {
        let info = ModuleInfo::analyse(text);
        if info.has_unrecognised_items {
            return false;
        }
        let unit = self.objects;
        self.objects += 1;
        let source = std::str::from_utf8(text).unwrap_or("");
        let blanked = source.contains('/').then(|| scan::blank_comments(source));
        let source = blanked.as_deref().unwrap_or(source);
        let defs = info
            .func_defs
            .iter()
            .map(|f| (f.name.as_str(), f.range.start..f.body.start));
        let decls = info.func_decls.iter().map(|f| (f.name.as_str(), f.range.clone()));
        let vars = info.variables.iter().map(|v| (v.name.as_str(), v.range.clone()));
        let declarations: Vec<_> = defs.chain(decls).chain(vars).collect();
        let shared: HashSet<_> = declarations
            .iter()
            .filter_map(|(name, range)| {
                let declaration = &source[range.clone()];
                let head = declaration.split('=').next().unwrap_or(declaration);
                head.split_ascii_whitespace()
                    .any(|word| matches!(word, ".extern" | ".visible" | ".weak" | ".common"))
                    .then_some(*name)
            })
            .collect();
        let mut symbols = HashMap::new();
        for (name, _) in declarations {
            let id = *self
                .nodes
                .entry(((!shared.contains(name)).then_some(unit), name.to_string()))
                .or_insert_with(|| {
                    self.callers.push(Vec::new());
                    self.callers.len() - 1
                });
            symbols.insert(name, id);
        }
        for f in &info.func_defs {
            let caller = symbols[f.name.as_str()];
            let functions = &info.func_refs[&f.name];
            let Some(items) = scan::items(source.as_bytes(), f.body.clone()) else {
                return false;
            };
            if !calls::used(source, &items).is_empty() {
                self.indirect.push((caller, f.name.clone()));
            }
            for target in functions.iter().chain(&info.var_refs[&f.name]) {
                self.callers[symbols[target.as_str()]].push(caller);
            }
        }
        // A definition can initialize a table which is declared extern and read from another unit.
        for var in &info.variables {
            if let Some((_, value)) = source[var.range.clone()].split_once('=') {
                let table = symbols[var.name.as_str()];
                for target in find_referenced_names(value, |name| symbols.contains_key(name)) {
                    self.callers[symbols[target]].push(table);
                }
            }
        }
        self.kernels
            .extend(info.entries.iter().map(|name| (symbols[name.as_str()], name.clone())));
        for trap in traps {
            if let Some(&id) = symbols.get(trap.function.as_str()) {
                self.trapped.push((id, trap.clone()));
            }
        }
        true
    }

    /// The kernels in the complete link that can reach one of its trapped functions.
    pub fn refused(mut self) -> Vec<crate::Refused> {
        let entries: HashSet<_> = self.kernels.iter().map(|(id, _)| *id).collect();
        if let Some((_, trap)) = self.trapped.iter().find(|(id, _)| !entries.contains(id)) {
            // A pointer passed in from the host need not appear in any PTX initializer. Only a
            // device function can be its target. Other kernels remain usable.
            let reason = format!("indirect call may reach {}: {}", trap.function, trap.reason);
            self.trapped.extend(self.indirect.into_iter().map(|(id, function)| {
                (
                    id,
                    crate::Trapped {
                        function,
                        reason: reason.clone(),
                    },
                )
            }));
        }
        let mut reasons = vec![None; self.callers.len()];
        let mut pending = Vec::new();
        for (i, (id, _)) in self.trapped.iter().enumerate() {
            reasons[*id] = Some(i);
            pending.push(*id);
        }
        while let Some(id) = pending.pop() {
            for &caller in &self.callers[id] {
                if reasons[caller].is_none() {
                    reasons[caller] = reasons[id];
                    pending.push(caller);
                }
            }
        }
        self.kernels
            .into_iter()
            .filter_map(|(id, kernel)| {
                let trap = &self.trapped[reasons[id]?].1;
                Some(crate::Refused {
                    kernel,
                    function: trap.function.clone(),
                    reason: trap.reason.clone(),
                })
            })
            .collect()
    }
}

/// Include a linkage qualifier before the directive, including across newlines. Comments were
/// already blanked when the offsets were determined.
fn find_def_start(text: &[u8], directive: usize) -> usize {
    let mut end = directive;
    while end > 0 && text[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    let mut start = end;
    while start > 0 && is_name_byte(text[start - 1]) {
        start -= 1;
    }
    if matches!(&text[start..end], b".visible" | b".weak" | b".extern") {
        start
    } else {
        directive
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_weak_leaves_match_the_executable_and_keep_their_debug_info() {
        fn module(linkage: &str, storage: &str, size: u32, value: u32, file: &str) -> ModuleInfo {
            let text = format!(
                ".version 8.0\n.target sm_70\n.address_size 64\n\
                 .file 1 \"{file}\"\n.weak {storage} .align 4 .b8 scratch[{size}];\n\
                 {linkage} .entry project(.param .u64 output) {{\n\
                 .reg .b32 a; .loc 1 3 0\n\
                 mov.u32 a, {value}; st.shared.b32 [scratch], a; ret;\n}}\n"
            );
            ModuleInfo::analyse(text.as_bytes())
        }
        let first = module(".weak", ".shared", 8, 4, "first.cu");
        let second = module(".weak", ".shared", 8, 4, "second.cu");
        let mut shared = LinkedShared::default();
        shared.add_object(&first.text).unwrap();
        shared.add_object(&second.text).unwrap();
        let key = first.weak_leaf_definition("project", &shared).unwrap();
        assert_eq!(second.weak_leaf_definition("project", &shared).unwrap(), key);
        let emitted = String::from_utf8(first.extract_linked_leaf("project", &shared).unwrap()).unwrap();
        assert!(emitted.contains("first.cu") && emitted.contains(".loc 1 3 0"));
        assert_ne!(
            module(".weak", ".shared", 8, 5, "first.cu")
                .weak_leaf_definition("project", &shared)
                .unwrap(),
            key
        );
        assert!(
            module(".visible", ".shared", 8, 4, "first.cu")
                .weak_leaf_definition("project", &shared)
                .is_none()
        );
        assert!(
            module(".weak", ".global", 8, 4, "first.cu")
                .weak_leaf_definition("project", &shared)
                .is_none()
        );
    }

    #[test]
    fn weak_entries_compare_the_targets_of_local_branches() {
        let first = r#".version 8.0
.target sm_70
.address_size 64
.file 1 "first.cu"
.weak .entry k(.param .u64 arg) {
    .reg .pred %p; .reg .u32 %r; .reg .u64 %out;
    .loc 1 4 0
    ld.param.u64 %out, [arg]; mov.u32 %r, %tid.x; setp.eq.u32 %p, %r, 1;
    .loc 1 9 0, function_name $L__debug1, inlined_at 1 4 0
    @%p bra $L__B3_a;
    bra.uni $L__B3_b;
$L__B3_a:
    { add.u32 %r, %r, 1; }
    bra $L__B3_b;
$L__B3_b:
    st.global.u32 [%out], %r; ret;
}

.section .debug_str
{
$L__debug1:
.b8 107
.b8 0
}
"#;
        let renamed = first
            .replace("B3_", "B81_")
            .replace("first.cu", "other.cu")
            .replace(".file 1 ", ".file 2 ")
            .replace(".loc 1 ", ".loc 2 ")
            .replace("inlined_at 1 ", "inlined_at 2 ")
            .replace("$L__debug1", "$L__debug33");
        fn key(s: &str) -> Vec<u8> {
            ModuleInfo::analyse(s.as_bytes())
                .weak_leaf_definition("k", &LinkedShared::default())
                .unwrap()
        }
        assert_eq!(key(first), key(&renamed));
        let same_line = first.replace("inlined_at 1 4 0\n    @%p bra", "inlined_at 1 4 0 @%p bra");
        assert_ne!(same_line, first);
        assert_eq!(key(first), key(&same_line));
        assert_ne!(key(first), key(&same_line.replace("@%p bra", "@!%p bra")));
        assert_eq!(
            key(first),
            key(&renamed.replace("    @%p", "    // different nvcc backend comment\n    @%p"))
        );
        let a = first.replace(".param .u64 arg", ".param .u64 arg, .param .u32 $L__B3_a");
        let b = renamed.replace(".param .u64 arg", ".param .u64 arg, .param .u32 $L__B3_a");
        assert_ne!(key(&a), key(&b));
        let a = first.replace("add.u32 %r, %r, 1", "mov.u32 %r, sym/* a */bol");
        let b = renamed.replace("add.u32 %r, %r, 1", "mov.u32 %r, symbol");
        assert_ne!(key(&a), key(&b));
        let emitted = ModuleInfo::analyse(first.as_bytes())
            .extract_linked_leaf("k", &LinkedShared::default())
            .unwrap();
        let emitted = String::from_utf8(emitted).unwrap();
        assert!(emitted.contains("@%p bra $L__B3_a;"));
        assert!(emitted.contains(".loc 1 9 0, function_name $L__debug1"));
        for wrong in [
            renamed.replacen("@%p bra $L__B81_a", "@%p bra $L__B81_b", 1),
            renamed.replace("@%p bra", "@!%p bra"),
            renamed.replace("bra.uni", "bra"),
            renamed.replace("add.u32 %r, %r, 1", "add.u32 %r, %r, 2"),
            renamed.replace("[arg]", "[other]"),
            renamed.replace("add.u32 %r, %r, 1", "add.u32 %r, %r, %s"),
        ] {
            assert_ne!(key(first), key(&wrong));
        }
        // A name with a second role or definition cannot be treated as a private branch target.
        for extra in [
            "mov.u64 %r, $L__B3_a;",
            ".reg .u32 $L__B3_a;",
            ".pragma \"$L__B3_a\";",
            "$L__B3_a: ret;",
            "{ $L__B3_a: ret; }",
        ] {
            let a = first.replace("    @%p", &format!("    {extra}\n    @%p"));
            let b = renamed.replace("    @%p", &format!("    {extra}\n    @%p"));
            assert_ne!(key(&a), key(&b), "{extra}");
        }
        // A pragma or instruction on a malformed debug line is never omitted from the key.
        let a = first.replace(".loc 1 9 0,", ".loc 1 9 0 add.u32 %r, %r, 5; ,");
        let b = renamed.replace(".loc 2 9 0,", ".loc 2 9 0 add.u32 %r, %r, 6; ,");
        assert_ne!(key(&a), key(&b));
        let a = first.replace("    @%p", "    .pragma \"keep.a\"; @%p");
        let b = renamed.replace("    @%p", "    .pragma \"keep.b\"; @%p");
        assert_ne!(key(&a), key(&b));
    }

    #[test]
    fn shared_bindings_include_objects_without_the_entry() {
        fn object(linkage: &str, alignment: u32, bytes: u32, entry: &str) -> String {
            format!(
                ".version 8.0\n.target sm_70\n.address_size 64\n\
                     {linkage} .shared .align {alignment} .b8 scratch[{bytes}];\n\
                     .weak .entry {entry}() {{ .reg .b32 r; ld.shared.b32 r, [scratch]; ret; }}\n"
            )
        }
        let weak = object(".weak", 4, 8, "project");
        let module = ModuleInfo::analyse(weak.as_bytes());
        for (linkage, align, bytes, allowed) in [
            (".weak", 4, 8, true),
            ("", 128, 16, true), // A different object's private storage cannot bind ours.
            (".visible", 4, 8, false),
            (".extern", 4, 8, false),
            (".weak", 128, 8, false),
            (".weak", 4, 16, false),
        ] {
            let other = object(linkage, align, bytes, "other");
            for inputs in [
                vec![&weak, &other],        // The unique entry case.
                vec![&weak, &weak, &other], // A third object defines a different entry.
                vec![&other, &weak, &weak], // Binding must be safe in either order.
            ] {
                let mut shared = LinkedShared::default();
                for input in inputs {
                    shared.add_object(input.as_bytes()).unwrap();
                }
                assert_eq!(module.extract_linked_leaf("project", &shared).is_some(), allowed);
                assert_eq!(module.weak_leaf_definition("project", &shared).is_some(), allowed);
            }
        }
        let private = object("", 4, 8, "project");
        let strong = object(".visible", 4, 8, "other");
        let mut shared = LinkedShared::default();
        shared.add_object(private.as_bytes()).unwrap();
        shared.add_object(strong.as_bytes()).unwrap();
        assert!(
            ModuleInfo::analyse(private.as_bytes())
                .extract_linked_leaf("project", &shared)
                .is_some()
        );
        assert!(
            ModuleInfo::analyse(strong.as_bytes())
                .extract_linked_leaf("other", &shared)
                .is_none()
        );
    }

    #[test]
    fn linked_dynamic_shared_accepts_only_matching_unsized_externs() {
        let header = ".version 8.0\n.target sm_70\n.address_size 64\n";
        let dynamic = ".extern .shared .align 16 .b8 scratch[];";
        let kernel = format!("{header}{dynamic}\n.weak .entry k() {{ .reg .u32 r; mov.u32 r, scratch; ret; }}\n");
        let module = ModuleInfo::analyse(kernel.as_bytes());
        for decl in [
            ".extern .shared .b8 scratch[4][];",
            ".extern .shared::cluster .b8 scratch[];",
        ] {
            assert!(!is_unsized_array(decl, "scratch"));
        }
        for (other, allowed) in [
            (dynamic, true),
            (".extern .shared .align 8 .b8 scratch[];", false),
            (".extern .shared .align 16 .b8 scratch[16];", false),
            (".visible .shared .align 16 .b8 scratch[16];", false),
            (".weak .shared .align 16 .b8 scratch[16];", false),
            (".visible .global .align 16 .b8 scratch[16];", false),
            (".shared .align 4 .b8 scratch[32];", true), // Private to the other object.
        ] {
            let other = format!("{header}{other}\n");
            for inputs in [[&kernel, &other], [&other, &kernel]] {
                let mut shared = LinkedShared::default();
                for input in inputs {
                    shared.add_object(input.as_bytes()).unwrap();
                }
                assert_eq!(module.extract_linked_leaf("k", &shared).is_some(), allowed);
                assert_eq!(module.weak_leaf_definition("k", &shared).is_some(), allowed);
            }
        }
    }

    #[test]
    fn module_pragmas_require_the_linker_and_function_pragmas_stay_with_the_leaf() {
        let header = ".version 8.0\n.target sm_70\n.address_size 64\n";
        let kernel = ".weak .entry k() { .pragma \"nounroll\"; ret; }\n";
        let source = format!("{header}{kernel}");
        let mut shared = LinkedShared::default();
        shared.add_object(source.as_bytes()).unwrap();
        let module = ModuleInfo::analyse(source.as_bytes());
        assert_eq!(module.extract_linked_leaf("k", &shared).unwrap(), source.as_bytes());
        for source in [
            format!("{header}.pragma \"nounroll\";\n{kernel}"),
            format!("{header}{kernel}.pragma\t\"nounroll\";\n"),
            format!("{header}.file 1 \"f.cu\" .pragma \"nounroll\";\n{kernel}"),
        ] {
            let module = ModuleInfo::analyse(source.as_bytes());
            assert!(module.has_module_pragma);
            assert!(!module.has_unrecognised_items);
            assert!(!module.is_entry_independent("k"));
            assert!(module.extract("k").is_none());
            assert!(module.weak_leaf_definition("k", &shared).is_none());
            assert!(module.extract_linked_leaf("k", &shared).is_none());
            assert!(LinkedShared::default().add_object(source.as_bytes()).is_some());
        }
    }

    #[test]
    fn same_line_shared_declarations_cannot_escape_the_index() {
        for directive in [".address_size 64", ".file 1 \".shared and .pragma inside a filename\""] {
            let shared = format!(".version 8.0\n.target sm_70\n{directive} .visible .shared .b8 scratch[8];\n");
            assert!(LinkedShared::default().add_object(shared.as_bytes()).is_none());
        }
        let filename = b".version 8.0\n.target sm_70\n.file 1 \".shared and .pragma\"\n.visible .entry k() { ret; }\n";
        let mut shared = LinkedShared::default();
        shared.add_object(filename).unwrap();
        assert!(
            ModuleInfo::analyse(filename)
                .extract_linked_leaf("k", &shared)
                .is_some()
        );
    }

    #[test]
    fn linked_leaf_keeps_nvcc_line_info_and_defers_calls_and_globals() {
        let source = br#".version 8.8
.target sm_70
.address_size 64
.extern .func helper();
.extern .global .u32 counter;
.visible .entry leaf() {
    .loc	1 2 0, function_name $str, inlined_at 1 3 1
    ret;
}
.visible .entry direct() { call helper, (); ret; }
.visible .entry indirect(.param .u64 ptr) {
    .reg .b64 %rd;
    ld.param.u64 %rd, [ptr];
    proto: .callprototype _ ();
    call %rd, (), proto;
    ret;
}
.visible .entry shared_value() { .reg .u32 %r; ld.global.u32 %r, [counter]; ret; }
.file	1 "linked.cu"
.section	.debug_str
{
$str:
.b8 97
.b8 0
}
"#;
        let info = super::ModuleInfo::analyse(source);
        let mut shared = LinkedShared::default();
        shared.add_object(source).unwrap();
        let text = String::from_utf8(info.extract_linked_leaf("leaf", &shared).unwrap()).unwrap();
        assert!(text.contains(".file\t1") && text.contains(".debug_str") && text.contains("$str:"));
        for dependent in ["direct", "indirect", "shared_value", "missing"] {
            assert!(info.extract_linked_leaf(dependent, &shared).is_none(), "{dependent}");
        }
    }

    #[test]
    fn linked_kernel_reading_a_table_is_left_to_the_linker() {
        // Several objects of one link can reach a table through each other (a pointer stored by one,
        // a call into another): a kernel of a linked library is never compiled alone with a copy of
        // a table, even one that the module alone would copy.
        let source = format!(
            "{}\n{}",
            TABLE,
            ".visible .entry reader(.param .u64 out) {\n\t.reg .b32 %r<2>;\n\t.reg .b64 %rd<3>;\n\
             \tld.param.u64 %rd1, [out];\n\tld.global.u32 %r1, [t];\n\tst.global.u32 [%rd1], %r1;\n\tret;\n}\n\
             .visible .entry other() { ret; }\n"
        );
        let source = format!(".version 8.8\n.target sm_70\n.address_size 64\n{source}");
        let info = super::ModuleInfo::analyse(source.as_bytes());
        assert_eq!(info.copied_tables("reader"), ["t"]);
        let mut shared = LinkedShared::default();
        shared.add_object(source.as_bytes()).unwrap();
        assert!(info.extract_linked_leaf("reader", &shared).is_none());
        assert!(info.extract_linked_leaf("other", &shared).is_some());
    }

    #[test]
    fn linked_calls_and_external_table_reach_a_helper_in_another_object() {
        let mut graph = super::LinkedDependencies::default();
        assert!(graph.add(
            br#".version 8.0
.target sm_70
.address_size 64
.extern
.func $Library();
.extern
.global .u64 table;
.func %private_helper() { ret; }
.visible .entry direct() { call $Library, (); ret; }
.visible .entry indirect() { .reg .u64 %r; ld.global.u64 %r, [table]; ret; }
.visible .entry unaffected() { call %private_helper, (); ret; }
"#,
            &[],
        ));
        assert!(graph.add(
            br#".version 8.0
.target sm_70
.address_size 64
.pragma "nounroll";
.func %private_helper() { trap; } .visible
.func $Library() { call %private_helper, (); ret; }
.visible .global .u64 table = $Library;
"#,
            &[crate::Trapped {
                function: "%private_helper".into(),
                reason: "fixture".into(),
            }],
        ));
        let refused = graph.refused();
        let names: Vec<_> = refused.iter().map(|r| r.kernel.as_str()).collect();
        assert_eq!(names, ["direct", "indirect"]);
        assert!(refused.iter().all(|r| r.function == "%private_helper"));
    }

    #[test]
    fn indirect_call_through_a_kernel_argument_may_reach_a_trapped_device_function() {
        let mut graph = LinkedDependencies::default();
        assert!(graph.add(
            br#".version 8.0
.target sm_70
.address_size 64
.func invoke(.reg .b64 fp) {
    proto: .callprototype _ ();
    call.uni fp, (), proto;
    ret;
}
.visible .entry dispatch(.param .u64 pointer) {
    .reg .b64 f;
    ld.param.u64 f, [pointer];
    call invoke, (f);
    ret;
}
.func good() { ret; }
.visible .entry bounded(.param .u64 pointer) {
    .reg .b64 f;
    ld.param.u64 f, [pointer];
    known: .calltargets good;
    unused: .callprototype _ ();
    call.uni f, known;
    ret;
}
.visible .entry safe() { ret; }
"#,
            &[]
        ));
        assert!(graph.add(
            b".version 8.0\n.target sm_70\n.address_size 64\n.visible .func bad() { trap; }\n",
            &[crate::Trapped {
                function: "bad".into(),
                reason: "fixture".into()
            }]
        ));
        let refused = graph.refused();
        assert_eq!(refused.len(), 1);
        assert_eq!(refused[0].kernel, "dispatch");
        assert!(refused[0].reason.contains("indirect call"));
    }

    #[test]
    fn unknown_link_directives_require_the_original_linker() {
        let mut graph = LinkedDependencies::default();
        assert!(!graph.add(
            b".version 8.0\n.target sm_70\n.address_size 64\n.alias alias, real;\n",
            &[]
        ));
    }

    const MODULE: &str = "\
.version 9.0
.target sm_70
.address_size 64

.global .align 4 .b8 table[4] = {1, 2, 3, 4};
.global .align 4 .b32 counter;

.extern .func (.param .b32 func_retval0) helper (.param .b32 a)
;
.func (.reg .b32 %ret) twice (.reg .b32 %x)
{
\tadd.s32 %ret, %x, %x;
\tret;
}

.func (.reg .b32 %ret) quad (.reg .b32 %x)
{
\tcall (%ret), twice, (%x);
\tcall (%ret), twice, (%ret);
\tret;
}

.visible .entry kernel_a(
\t.param .u64 kernel_a_param_0
)
{
\t.reg .b32 %r<4>;
\tld.global.b32 %r1, [table];
\tcall (%r2), twice, (%r1);
\tst.global.b32 [kernel_a_param_0], %r2;
\tret;
}

.visible .entry kernel_b(
\t.param .u64 kernel_b_param_0
)
{
\t.reg .b32 %r<4>;
\tld.global.b32 %r1, [table];
\tcall (%r2), quad, (%r1);
\tst.global.b32 [kernel_b_param_0], %r2;
\tret;
}

.visible .entry kernel_c(
\t.param .u64 kernel_c_param_0
)
{
\t.reg .b32 %r<4>;
\tatom.global.add.s32 %r1, [counter], 1;
\tret;
}
";

    #[test]
    fn analyses_module_structure() {
        let info = ModuleInfo::analyse(MODULE.as_bytes());
        assert_eq!(info.entries, ["kernel_a", "kernel_b", "kernel_c"]);
        let func_names: Vec<&str> = info
            .func_defs
            .iter()
            .filter(|f| !f.is_entry)
            .map(|f| f.name.as_str())
            .collect();
        assert!(func_names.contains(&"twice"));
        assert!(func_names.contains(&"quad"));
    }

    #[test]
    fn finds_function_calls() {
        let info = ModuleInfo::analyse(MODULE.as_bytes());
        assert!(info.func_refs["kernel_a"].contains("twice"));
        assert!(!info.func_refs["kernel_a"].contains("quad"));
        assert!(info.func_refs["kernel_b"].contains("quad"));
        assert!(info.func_refs["quad"].contains("twice"));
    }

    #[test]
    fn finds_variable_references() {
        let info = ModuleInfo::analyse(MODULE.as_bytes());
        assert!(info.var_refs["kernel_a"].contains("table"));
        assert!(info.var_refs["kernel_b"].contains("table"));
        assert!(info.var_refs["kernel_c"].contains("counter"));
    }

    #[test]
    fn detects_written_variables() {
        let info = ModuleInfo::analyse(MODULE.as_bytes());
        assert!(info.written_vars.contains("counter"));
        assert!(!info.written_vars.contains("table"));
    }

    #[test]
    fn splittable_when_no_persistent_vars() {
        // A module with only .shared variables (per-launch) and no .global/.const is splittable.
        let text = "\
.version 9.0
.target sm_70
.address_size 64

.extern .shared .align 4 .b8 smem[];

.visible .entry k1() { .reg .u32 %r<2>; mov.u32 %r1, smem; ret; }
.visible .entry k2() { .reg .u32 %r<2>; mov.u32 %r1, smem; ret; }
";
        let info = ModuleInfo::analyse(text.as_bytes());
        assert_eq!(info.splittable(), Splittable::Yes);
    }

    #[test]
    fn module_with_persistent_vars_is_splittable_at_module_level() {
        // MODULE has .global variables referenced by entries. splittable() allows it at the
        // module level because per-entry classification handles the persistence constraint:
        // entries that reference persistent vars are dependent and come from the whole module.
        let info = ModuleInfo::analyse(MODULE.as_bytes());
        assert_eq!(info.splittable(), Splittable::Yes);
    }

    #[test]
    fn entries_reading_a_table_are_independent_and_a_writer_is_not() {
        // MODULE: kernel_a and kernel_b load from `table` (a .global with an initializer that no
        // function writes): each can have a copy of its own. kernel_c atomically writes `counter`,
        // so it is dependent. (Until 2026-11-23 every load from a persistent variable made its
        // kernel dependent.)
        let info = ModuleInfo::analyse(MODULE.as_bytes());
        assert!(info.is_entry_independent("kernel_a"), "kernel_a only reads table");
        assert!(info.is_entry_independent("kernel_b"), "kernel_b only reads table");
        assert_eq!(info.copied_tables("kernel_a"), ["table"]);
        assert!(
            !info.is_entry_independent("kernel_c"),
            "kernel_c writes counter, should be dependent"
        );
        assert!(info.copied_tables("kernel_c").is_empty());
        assert_eq!(info.dependent_vars("kernel_c"), ["counter"]);
        assert!(info.dependent_vars("kernel_a").is_empty());
        let sub = String::from_utf8(info.extract("kernel_a").unwrap()).unwrap();
        assert!(sub.contains(".global .align 4 .b8 table[4] = {1, 2, 3, 4};"), "{sub}");
        assert!(!sub.contains("counter") && !sub.contains("kernel_b"), "{sub}");
    }

    #[test]
    fn not_splittable_with_shared_mutable() {
        let text = MODULE.replace(
            "atom.global.add.s32 %r1, [counter], 1;",
            "ld.global.b32 %r1, [counter];",
        );
        let text = text.replace(
            "st.global.b32 [kernel_a_param_0], %r2;",
            "st.global.b32 [kernel_a_param_0], %r2;\n\tatom.global.add.s32 %r1, [counter], 1;",
        );
        let info = ModuleInfo::analyse(text.as_bytes());
        match info.splittable() {
            Splittable::No { reason } => {
                assert!(reason.contains("counter"), "reason: {reason}");
            }
            Splittable::Yes => panic!("should not be splittable"),
        }
    }

    #[test]
    fn taint_analysis_detects_indirect_write() {
        // nvcc pattern: mov address, cvta, then atom through the register.
        let text = "\
.version 9.0
.target sm_70
.address_size 64

.global .align 4 .b32 counter;

.visible .entry writer(
\t.param .u64 writer_param_0
)
{
\t.reg .u64 %rd<4>;
\t.reg .u32 %r<2>;
\tmov.u64 %rd1, counter;
\tcvta.to.global.u64 %rd2, %rd1;
\tatom.global.add.u32 %r1, [%rd2], 1;
\tret;
}

.visible .entry reader(
\t.param .u64 reader_param_0
)
{
\t.reg .u64 %rd<4>;
\t.reg .u32 %r<2>;
\tmov.u64 %rd1, counter;
\tcvta.to.global.u64 %rd2, %rd1;
\tld.global.u32 %r1, [%rd2];
\tst.global.u32 [reader_param_0], %r1;
\tret;
}
";
        let info = ModuleInfo::analyse(text.as_bytes());
        assert!(
            info.written_vars.contains("counter"),
            "counter should be detected as written"
        );
        match info.splittable() {
            Splittable::No { reason } => {
                assert!(reason.contains("counter"), "reason: {reason}");
            }
            Splittable::Yes => panic!("should not be splittable: counter is written by writer and read by reader"),
        }
    }

    #[test]
    fn entry_loading_visible_global_is_dependent() {
        // k1 loads from host_var (a .visible .global), so it is dependent. k2 does not
        // reference any persistent variable, so it is independent.
        let text = "\
.version 9.0
.target sm_70
.address_size 64

.visible .global .align 4 .b32 host_var;

.visible .entry k1() { .reg .u32 %r<2>; ld.global.u32 %r1, [host_var]; ret; }
.visible .entry k2() { ret; }
";
        let info = ModuleInfo::analyse(text.as_bytes());
        assert!(
            !info.is_entry_independent("k1"),
            "k1 loads from host_var, should be dependent"
        );
        assert!(
            info.is_entry_independent("k2"),
            "k2 uses no persistent var, should be independent"
        );
    }

    #[test]
    fn visible_global_with_linkage_prefix_is_parsed() {
        let text = "\
.version 9.0
.target sm_70
.address_size 64

.visible .global .align 4 .b32 vis_var = 42;
.weak .global .align 4 .b32 weak_var;

.visible .entry k() { .reg .u32 %r<2>; ld.global.u32 %r1, [vis_var]; ret; }
.visible .entry k2() { ret; }
";
        let info = ModuleInfo::analyse(text.as_bytes());
        let var_names: Vec<&str> = info.variables.iter().map(|v| v.name.as_str()).collect();
        assert!(var_names.contains(&"vis_var"), "vis_var not found: {var_names:?}");
        assert!(var_names.contains(&"weak_var"), "weak_var not found: {var_names:?}");
        assert_eq!(
            info.variables.iter().find(|v| v.name == "vis_var").unwrap().linkage,
            Linkage::External
        );
    }

    #[test]
    fn function_address_taken_by_mov_is_a_dependency() {
        let text = "\
.version 9.0
.target sm_70
.address_size 64

.func helper() { ret; }

.visible .entry k(
\t.param .u64 k_param_0
)
{
\t.reg .u64 %rd<2>;
\tld.param.u64 %rd0, [k_param_0];
\tmov.u64 %rd1, helper;
\tst.global.u64 [%rd0], %rd1;
\tret;
}
";
        let info = ModuleInfo::analyse(text.as_bytes());
        // `helper` should be found as a reference of `k` even without a direct call.
        let sub = info.extract("k").unwrap();
        let sub_text = std::str::from_utf8(&sub).unwrap();
        assert!(sub_text.contains("helper"), "helper should be in the sub-module");
    }

    #[test]
    fn unrecognised_module_level_item_prevents_splitting() {
        let text = "\
.version 9.0
.target sm_70
.address_size 64

.some_unknown_directive 42;

.visible .entry k1() { ret; }
.visible .entry k2() { ret; }
";
        let info = ModuleInfo::analyse(text.as_bytes());
        assert!(info.has_unrecognised_items);
        match info.splittable() {
            Splittable::No { reason } => {
                assert!(reason.contains("unrecognised"), "reason: {reason}");
            }
            Splittable::Yes => panic!("should not be splittable with unrecognised items"),
        }
    }

    #[test]
    fn extracts_submodule_for_kernel_a() {
        let info = ModuleInfo::analyse(MODULE.as_bytes());
        let sub = info.extract("kernel_a").unwrap();
        let sub_text = std::str::from_utf8(&sub).unwrap();
        assert!(sub_text.contains(".version 9.0"));
        assert!(sub_text.contains(".target sm_70"));
        assert!(sub_text.contains(".visible .entry kernel_a"));
        assert!(sub_text.contains("twice"));
        assert!(sub_text.contains("table[4]"));
        assert!(!sub_text.contains("kernel_b"));
        assert!(!sub_text.contains("kernel_c"));
        assert!(!sub_text.contains("counter"));
    }

    #[test]
    fn extracts_submodule_for_kernel_b() {
        let info = ModuleInfo::analyse(MODULE.as_bytes());
        let sub = info.extract("kernel_b").unwrap();
        let sub_text = std::str::from_utf8(&sub).unwrap();
        assert!(sub_text.contains(".visible .entry kernel_b"));
        assert!(sub_text.contains("quad"));
        assert!(sub_text.contains("twice"));
        assert!(sub_text.contains("table[4]"));
        assert!(!sub_text.contains("kernel_a"));
        assert!(!sub_text.contains("kernel_c"));
        assert!(!sub_text.contains("counter"));
    }

    #[test]
    fn extracts_submodule_for_kernel_c() {
        let info = ModuleInfo::analyse(MODULE.as_bytes());
        let sub = info.extract("kernel_c").unwrap();
        let sub_text = std::str::from_utf8(&sub).unwrap();
        assert!(sub_text.contains(".visible .entry kernel_c"));
        assert!(sub_text.contains("counter"));
        assert!(!sub_text.contains("twice"));
        assert!(!sub_text.contains("table"));
        assert!(!sub_text.contains("kernel_a"));
        assert!(!sub_text.contains("kernel_b"));
    }

    #[test]
    fn extract_returns_none_for_unknown_kernel() {
        let info = ModuleInfo::analyse(MODULE.as_bytes());
        assert!(info.extract("nonexistent").is_none());
    }

    #[test]
    fn single_entry_module_is_splittable() {
        let text = b".version 9.0\n.target sm_70\n.entry only()\n{\n\tret;\n}\n";
        let info = ModuleInfo::analyse(text);
        assert_eq!(info.entries, ["only"]);
        assert_eq!(info.splittable(), Splittable::Yes);
    }

    #[test]
    fn extracted_submodule_has_valid_header() {
        let info = ModuleInfo::analyse(MODULE.as_bytes());
        for entry in &info.entries {
            let sub = info.extract(entry).unwrap();
            let header = crate::header(&sub);
            assert!(header.is_some(), "sub-module for {entry} has no header");
            let funcs = crate::functions(&sub);
            assert!(
                funcs
                    .iter()
                    .any(|f| { std::str::from_utf8(&sub[f.name.clone()]).unwrap() == entry }),
                "sub-module for {entry} does not contain its entry"
            );
        }
    }

    #[test]
    fn comments_in_instructions_do_not_hide_access_to_module_state() {
        let module = "\
.version 7.0
.target sm_75
.address_size 64
.global .u32 g;
.visible .entry access(.param .u64 output) {
    .reg .b64 %p, %out;
    .reg .b32 %value;
    ld.param.u64 %out, [output];
    mov.u64 %p, g;
    ld.global.u32 %value, [%p];
    st.global.u32 [%out], %value;
    ret;
}
.visible .entry unrelated() { ret; }
";
        for (original, commented) in [
            ("mov.u64 %p", "mov.u64 /* comma, */ %p"),
            ("mov.u64 %p", "mov.u64 // ; comma, [\n %p"),
            ("mov.u64 %p", "mov.u64/* %ignored, */%p"),
            ("ld.global.u32 %value,", "ld.global.u32 %value /* [ */,"),
            ("ld.global.u32 %value", "ld.global.u32/*words*/%value"),
        ] {
            for program in [module.to_owned(), module.replace(original, commented)] {
                let info = ModuleInfo::analyse(program.as_bytes());
                assert_eq!(info.dependent_vars("access"), ["g"], "{commented}");
                assert!(info.is_entry_independent("unrelated"));
            }
        }
    }

    #[test]
    fn names_in_comments_do_not_change_call_dependencies() {
        let module = "\
.version 7.0
.target sm_75
.address_size 64
.global .u32 g;
.extern .func (.param .b32 result) vprintf(.param .b64 fmt, .param .b64 args);
.func sink(.reg .b64 ptr) { ret; }
.visible .entry caller() {
    .reg .b64 %p;
    mov.u64 %p, g;
    call.uni sink /* vprintf */, (%p);
    ret;
}
.visible .entry unrelated() { ret; }
";
        let info = ModuleInfo::analyse(module.as_bytes());
        assert_eq!(info.dependent_vars("caller"), ["g"]);
        assert!(info.is_entry_independent("unrelated"));
        assert!(!info.func_refs["caller"].contains("vprintf"));
    }

    #[test]
    fn taint_through_brace_enclosed_statements() {
        // Compiler output puts statements on one line with braces:
        // `{ .reg .b64 %tmp; cvta.shared.u64 %rd12, %tmp; }`
        // The scanner must handle this; a line-based parser would miss the cvta statement.
        let text = "\
.version 9.0
.target sm_70
.address_size 64

.global .align 4 .b32 counter;

.visible .entry k()
{
\t.reg .u64 %rd<4>;
\t.reg .u32 %r<2>;
\t{ .reg .u64 %tmp; mov.u64 %tmp, counter; cvta.to.global.u64 %rd2, %tmp; }
\tatom.global.add.u32 %r1, [%rd2], 1;
\tret;
}

.visible .entry k2() { ret; }
";
        let info = ModuleInfo::analyse(text.as_bytes());
        assert!(
            info.written_vars.contains("counter"),
            "counter should be detected as written through brace-enclosed statements"
        );
    }

    #[test]
    fn taint_two_statements_one_line() {
        let text = "\
.version 9.0
.target sm_70
.address_size 64

.global .align 4 .b32 g;

.visible .entry k()
{
\t.reg .u64 %rd<4>; .reg .u32 %r<2>;
\tmov.u64 %rd1, g; cvta.to.global.u64 %rd2, %rd1;
\tst.global.u32 [%rd2], %r1;
\tret;
}

.visible .entry k2() { ret; }
";
        let info = ModuleInfo::analyse(text.as_bytes());
        assert!(info.written_vars.contains("g"), "g should be detected as written");
    }

    #[test]
    fn taint_converges_on_long_reverse_chain() {
        // A chain of 30 register copies in reverse order: %rd30 <- ... <- %rd1 <- counter.
        // The old 20-round cap would fail to propagate taint to %rd30.
        let mut body = String::from(
            ".version 9.0\n.target sm_70\n.address_size 64\n\n\
             .global .align 4 .b32 counter;\n\n\
             .visible .entry writer()\n{\n\t.reg .u64 %rd<32>;\n\t.reg .u32 %r<2>;\n",
        );
        // Write the chain in reverse: %rd30 = %rd29; %rd29 = %rd28; ... %rd1 = counter.
        for i in (1..=30).rev() {
            if i == 1 {
                body += "\tmov.u64 %rd1, counter;\n";
            } else {
                body += &format!("\tmov.u64 %rd{i}, %rd{};\n", i - 1);
            }
        }
        body += "\tatom.global.add.u32 %r1, [%rd30], 1;\n\tret;\n}\n\n";
        body += ".visible .entry reader() { ret; }\n";

        let info = ModuleInfo::analyse(body.as_bytes());
        assert!(
            info.written_vars.contains("counter"),
            "counter should be detected as written through a 30-register chain"
        );
    }

    #[test]
    fn dependencies_follow_initializer_chains_and_cycles() {
        // caller -> dispatch -> targets -> target -> dispatch. A function
        // address in a PTX initializer must already have been declared.
        let module = "\
.version 8.0
.target sm_90
.address_size 64
.func target();
.const .align 8 .u64 targets[1] /* ; = */ = {target};
.const .align 8 .u64 dispatch = targets;
.func target() {
    .reg .u32 %id;
    .reg .u64 %p;
    ld.const.u64 %p, [dispatch];
    mov.u32 %id, %cluster_ctaid.x;
    ret;
}
.visible .entry caller() {
    .reg .u64 %p, %fn;
    proto: .callprototype _ ();
    ld.const.u64 %p, [dispatch];
    ld.const.u64 %fn, [%p];
    call.uni %fn, (), proto;
    ret;
}
.visible .entry unrelated() { ret; }
";
        let info = ModuleInfo::analyse(module.as_bytes());
        let reached: HashMap<_, _> = info.reach().collect();
        assert_eq!(reached["caller"], ["caller", "target"]);
        assert_eq!(reached["unrelated"], ["unrelated"]);
        let caller = String::from_utf8(info.extract("caller").unwrap()).unwrap();
        assert_eq!(caller, module.split_once(".visible .entry unrelated").unwrap().0);
        let unrelated = String::from_utf8(info.extract("unrelated").unwrap()).unwrap();
        assert!(!unrelated.contains("targets"));
        assert!(!unrelated.contains(".func target"));

        let (_, report) = crate::lower_to_volta(module.as_bytes());
        assert_eq!(report.refused.len(), 1);
        assert_eq!(report.refused[0].kernel, "caller");
        assert_eq!(report.refused[0].function, "target");
    }

    #[test]
    fn variable_initializer_cycle_closes_and_preserves_forward_declarations() {
        let module = "\
.version 8.0
.target sm_90
.address_size 64
.extern .global .u64 second;
.global .u64 first = generic(second);
.visible .global .u64 second = generic(first);
.visible .entry caller() {
    .reg .u64 %p;
    ld.global.u64 %p, [first];
    ret;
}
.visible .entry unrelated() { ret; }
";
        let info = ModuleInfo::analyse(module.as_bytes());
        let mut vars = info.dependent_vars("caller");
        vars.sort();
        assert_eq!(vars, ["first", "second"]);
        assert!(info.dependent_vars("unrelated").is_empty());
        let caller = String::from_utf8(info.extract("caller").unwrap()).unwrap();
        assert_eq!(caller, module.split_once(".visible .entry unrelated").unwrap().0);
    }

    #[test]
    fn variable_in_initializer_is_escaped() {
        // `.global .u64 table[1] = {counter};` makes counter's address escape.
        // A load from table and a store through the loaded pointer should mark counter as written.
        let text = "\
.version 9.0
.target sm_70
.address_size 64

.global .align 4 .b32 counter;
.global .align 8 .u64 table[1] = {counter};

.visible .entry k1()
{
\t.reg .u64 %rd<4>;
\t.reg .u32 %r<2>;
\tld.global.u64 %rd1, [table];
\tst.global.u32 [%rd1], %r1;
\tret;
}

.visible .entry k2()
{
\t.reg .u32 %r<2>;
\tld.global.u32 %r1, [counter];
\tret;
}
";
        let info = ModuleInfo::analyse(text.as_bytes());
        assert!(
            info.written_vars.contains("counter"),
            "counter should be marked as escaped (its name appears in table's initializer)"
        );
        match info.splittable() {
            Splittable::No { reason } => {
                assert!(reason.contains("counter"), "reason: {reason}");
            }
            Splittable::Yes => panic!("should not be splittable: counter escapes through table"),
        }
    }

    #[test]
    fn vprintf_does_not_count_as_write() {
        let text = "\
.version 9.0
.target sm_70
.address_size 64

.global .align 1 .b8 args[8];
.extern .func (.param .b32 retval0) vprintf (.param .b64 a);

.visible .entry k()
{
\t.reg .b64 %rd<4>;
\t.reg .b32 %r<2>;
\tmov.u64 %rd1, args;
\tcall.uni (%r1), vprintf, (%rd1);
\tret;
}
";
        let info = ModuleInfo::analyse(text.as_bytes());
        assert!(!info.written_vars.contains("args"), "{:?}", info.written_vars);
    }

    #[test]
    fn module_defined_vprintf_gets_no_exception() {
        let text = "\
.version 9.0
.target sm_70
.address_size 64

.global .align 1 .b8 args[8];

.func (.param .b32 retval0) vprintf (.param .b64 a)
{
\tret;
}

.visible .entry k()
{
\t.reg .b64 %rd<4>;
\t.reg .b32 %r<2>;
\tmov.u64 %rd1, args;
\tcall.uni (%r1), vprintf, (%rd1);
\tret;
}
";
        let info = ModuleInfo::analyse(text.as_bytes());
        assert!(info.written_vars.contains("args"), "{:?}", info.written_vars);
    }

    #[test]
    fn tainted_arg_to_other_extern_counts_as_written() {
        let text = "\
.version 9.0
.target sm_70
.address_size 64

.global .align 1 .b8 args[8];
.extern .func (.param .b32 retval0) other_fn (.param .b64 a);

.visible .entry k()
{
\t.reg .b64 %rd<4>;
\t.reg .b32 %r<2>;
\tmov.u64 %rd1, args;
\tcall.uni (%r1), other_fn, (%rd1);
\tret;
}
";
        let info = ModuleInfo::analyse(text.as_bytes());
        assert!(info.written_vars.contains("args"), "{:?}", info.written_vars);
    }

    // ---- is_message_string_only / per-entry independence tests ----

    #[test]
    fn string_only_passed_to_vprintf_is_independent() {
        // A format string whose address is only passed to vprintf: the entry is independent.
        let text = "\
.version 9.0
.target sm_70
.address_size 64

.global .align 1 .b8 $str[6] = {72, 101, 108, 108, 111, 0};
.extern .func (.param .b32 retval0) vprintf (.param .b64 fmt, .param .b64 valist);

.visible .entry printer()
{
\t.reg .b64 %rd<4>;
\t.reg .b32 %r<2>;
\tmov.u64 %rd1, $str;
\tmov.u64 %rd2, 0;
\tcall.uni (%r1), vprintf, (%rd1, %rd2);
\tret;
}

.visible .entry noop() { ret; }
";
        let info = ModuleInfo::analyse(text.as_bytes());
        assert!(
            info.is_entry_independent("printer"),
            "printer only passes $str to vprintf, should be independent"
        );
        assert!(
            info.is_entry_independent("noop"),
            "noop references nothing, should be independent"
        );
    }

    #[test]
    fn string_also_loaded_from_is_dependent() {
        // The same string is passed to vprintf AND also loaded from: the entry is dependent.
        let text = "\
.version 9.0
.target sm_70
.address_size 64

.global .align 1 .b8 $str[6] = {72, 101, 108, 108, 111, 0};
.extern .func (.param .b32 retval0) vprintf (.param .b64 fmt, .param .b64 valist);

.visible .entry loader()
{
\t.reg .b64 %rd<4>;
\t.reg .b32 %r<2>;
\tmov.u64 %rd1, $str;
\tcall.uni (%r1), vprintf, (%rd1, %rd1);
\tld.global.u8 %r1, [%rd1];
\tret;
}

.visible .entry noop() { ret; }
";
        let info = ModuleInfo::analyse(text.as_bytes());
        assert!(
            !info.is_entry_independent("loader"),
            "loader loads from $str, should be dependent"
        );
    }

    /// A module with one table, `t`, read by `reader` and touched by `other` as `OTHER` says.
    fn table_module(declaration: &str, other: &str) -> String {
        format!(
            "\
.version 9.0
.target sm_70
.address_size 64

{declaration}

.visible .entry reader(
\t.param .u64 reader_param_0
)
{{
\t.reg .b32 %r<3>;
\t.reg .b64 %rd<6>;
\tld.param.u64 %rd1, [reader_param_0];
\tcvta.to.global.u64 %rd2, %rd1;
\tmov.u64 %rd3, t;
\tmov.u32 %r1, %tid.x;
\tmul.wide.u32 %rd4, %r1, 4;
\tadd.s64 %rd5, %rd3, %rd4;
\tld.global.nc.u32 %r2, [%rd5];
\tst.global.u32 [%rd2], %r2;
\tret;
}}

.visible .entry other()
{{
\t.reg .b32 %r<3>;
\t.reg .b64 %rd<4>;
{other}
\tret;
}}
"
        )
    }

    const TABLE: &str = ".global .align 4 .b8 t[8] = {1, 0, 0, 0, 2, 0, 0, 0};";

    #[test]
    fn lookup_table_loaded_from_is_copied() {
        // An entry loads from a .global lookup table that nothing writes: the entry is independent
        // and its sub-module holds the table, with its bytes. (Until 2026-11-23: dependent.)
        let text = table_module(TABLE, "\tld.global.u32 %r1, [t+4];");
        let info = ModuleInfo::analyse(text.as_bytes());
        assert!(info.is_entry_independent("reader"));
        assert!(info.is_entry_independent("other"));
        assert_eq!(info.copied_tables("reader"), ["t"]);
        let sub = String::from_utf8(info.extract("reader").unwrap()).unwrap();
        assert!(sub.contains(TABLE), "{sub}");
        assert!(!sub.contains(".entry other"), "{sub}");
        // The same in the constant space.
        let constant = ".const .align 4 .b8 t[8] = {1, 0, 0, 0, 2, 0, 0, 0};";
        let text = table_module(constant, "\tld.const.u32 %r1, [t];");
        let info = ModuleInfo::analyse(text.replace("ld.global.nc", "ld.const").as_bytes());
        assert!(info.is_entry_independent("reader"));
        assert_eq!(info.copied_tables("other"), ["t"]);
    }

    #[test]
    fn table_that_can_change_is_not_copied() {
        // Each of these makes `t` a variable whose copies could come to differ, or one whose bytes
        // the module does not give: `reader`, which only reads it, is then dependent.
        let cases = [
            ("a store into it", TABLE, "\tst.global.u32 [t+4], %r1;"),
            ("an atomic on it", TABLE, "\tatom.global.add.u32 %r1, [t], 1;"),
            (
                "a store through an address made from it",
                TABLE,
                "\tmov.u64 %rd1, t;\n\tadd.s64 %rd2, %rd1, 4;\n\tst.global.u32 [%rd2], %r1;",
            ),
            (
                "its address kept in local memory and loaded back",
                TABLE,
                "\tmov.u64 %rd1, t;\n\tst.local.u64 [%rd3], %rd1;\n\tld.local.u64 %rd2, [%rd3];\n\tst.global.u32 [%rd2], %r1;",
            ),
            (
                "its address stored as data",
                TABLE,
                "\tmov.u64 %rd1, t;\n\tst.global.u64 [%rd3], %rd1;",
            ),
            (
                "an asynchronous copy from it",
                TABLE,
                "\tmov.u64 %rd1, t;\n\tcp.async.ca.shared.global [%r1], [%rd1], 4;",
            ),
            (
                "no initializer",
                ".global .align 4 .b8 t[8];",
                "\tld.global.u32 %r1, [t];",
            ),
            (
                "a visible name",
                ".visible .global .align 4 .b8 t[8] = {1, 0, 0, 0, 2, 0, 0, 0};",
                "\tld.global.u32 %r1, [t];",
            ),
            (
                "a weak name",
                ".weak .global .align 4 .b8 t[8] = {1, 0, 0, 0, 2, 0, 0, 0};",
                "\tld.global.u32 %r1, [t];",
            ),
            (
                "managed memory",
                ".global .attribute(.managed) .align 4 .b8 t[8] = {1, 0, 0, 0, 2, 0, 0, 0};",
                "\tld.global.u32 %r1, [t];",
            ),
            (
                "an initializer that names a function",
                ".func f()\n{\n\tret;\n}\n.global .align 8 .u64 t[1] = {f};",
                "\tld.global.u64 %rd1, [t];",
            ),
            (
                "its address in another variable",
                ".global .align 4 .b8 t[8] = {1, 0, 0, 0, 2, 0, 0, 0};\n.global .align 8 .u64 p = generic(t);",
                "\tld.global.u64 %rd1, [p];",
            ),
            // A register needs no `%` (2026-11-23: these five changed `t` and left it copied).
            (
                "a store through a register whose name has no %",
                TABLE,
                "\t.reg .b64 p;\n\tmov.u64 p, t;\n\tst.global.u32 [p], %r1;",
            ),
            (
                "a store through an address made from it in a register whose name has no %",
                TABLE,
                "\t.reg .b64 p;\n\tmov.u64 %rd1, t;\n\tadd.s64 p, %rd1, 4;\n\tst.global.u32 [p], %r1;",
            ),
            (
                "its address passed to a call in a register whose name has no %",
                ".global .align 4 .b8 t[8] = {1, 0, 0, 0, 2, 0, 0, 0};\n.func f(\n\t.reg .b64 a\n)\n{\n\tst.global.u32 [a], 9;\n\tret;\n}",
                "\t.reg .b64 q;\n\tmov.u64 q, t;\n\tcall f, (q);",
            ),
            (
                "its address given back by a callee in a register of its signature",
                ".global .align 4 .b8 t[8] = {1, 0, 0, 0, 2, 0, 0, 0};\n.func (.reg .b64 rv) get()\n{\n\tmov.u64 rv, t;\n\tret;\n}",
                "\tcall (%rd1), get, ();\n\tst.global.u32 [%rd1], %r1;",
            ),
            (
                "its address moved into a parameter register of a function",
                ".global .align 4 .b8 t[8] = {1, 0, 0, 0, 2, 0, 0, 0};\n.func put(\n\t.reg .b64 a\n)\n{\n\tmov.u64 a, t;\n\tret;\n}",
                "\tld.global.u32 %r1, [t];",
            ),
            // An opcode needs no space before its operands, and a guard none after its `@`.
            (
                "a store written against its operands",
                TABLE,
                "\tst.global.u32[t], %r1;",
            ),
            (
                "its address taken by an instruction written against its operands",
                TABLE,
                "\tmov.u64%rd1, t;\n\tst.global.u32 [%rd1], %r1;",
            ),
            (
                "a store under a guard written with a space",
                TABLE,
                "\t.reg .pred %p<2>;\n\tsetp.eq.u32 %p1, %r1, 0;\n\t@ %p1 st.global.u32 [t], %r1;",
            ),
            (
                "its address taken under a guard written with spaces",
                TABLE,
                "\t.reg .pred %p<2>;\n\tsetp.eq.u32 %p1, %r1, 0;\n\t@ ! %p1 mov.u64 %rd1, t;\n\tst.global.u32 [%rd1], %r1;",
            ),
            // Other variables and declarations that hold its address.
            (
                "its address in a table of addresses",
                ".global .align 4 .b8 t[8] = {1, 0, 0, 0, 2, 0, 0, 0};\n.global .align 8 .u64 t2[1] = {t};",
                "\tld.global.u64 %rd1, [t2];\n\tst.u32 [%rd1], %r1;",
            ),
            (
                "an address made from it in a table of addresses",
                ".global .align 4 .b8 t[8] = {1, 0, 0, 0, 2, 0, 0, 0};\n.global .align 8 .u64 t2[1] = {generic(t)+4};",
                "\tld.global.u64 %rd1, [t2];\n\tst.u32 [%rd1], %r1;",
            ),
            (
                "a variable of the function initialized with its address",
                TABLE,
                "\t.global .align 8 .u64 q = generic(t);\n\tld.global.u64 %rd1, [q];\n\tst.u32 [%rd1], %r1;",
            ),
            (
                "another attribute",
                ".global .attribute(.unified(19,95)) .align 4 .b8 t[8] = {1, 0, 0, 0, 2, 0, 0, 0};",
                "\tld.global.u32 %r1, [t];",
            ),
            // Its address through memory, in parameter space, and in every other kind of write.
            (
                "its address kept in shared memory and loaded back",
                ".global .align 4 .b8 t[8] = {1, 0, 0, 0, 2, 0, 0, 0};\n.shared .align 8 .b8 s[8];",
                "\tmov.u64 %rd1, t;\n\tmov.u64 %rd3, s;\n\tst.shared.u64 [%rd3], %rd1;\n\tld.shared.u64 %rd2, [%rd3];\n\tst.global.u32 [%rd2], %r1;",
            ),
            (
                "its address exchanged by an atomic",
                TABLE,
                "\tmov.u64 %rd1, t;\n\tatom.global.exch.b64 %rd2, [%rd3], %rd1;",
            ),
            (
                "its address passed to a call in parameter space",
                ".global .align 4 .b8 t[8] = {1, 0, 0, 0, 2, 0, 0, 0};\n.func g(\n\t.param .b64 a\n)\n{\n\tret;\n}",
                "\t{\n\t.param .b64 param0;\n\tmov.u64 %rd1, t;\n\tst.param.b64 [param0], %rd1;\n\tcall.uni g, (param0);\n\t}",
            ),
            ("a reduction on it", TABLE, "\tred.global.add.u32 [t], 1;"),
            ("a volatile store into it", TABLE, "\tst.volatile.global.u32 [t], %r1;"),
            (
                "a store through its generic address",
                TABLE,
                "\tcvta.global.u64 %rd1, t;\n\tst.u32 [%rd1], %r1;",
            ),
            (
                "a store through selections and moves whose registers have no %",
                TABLE,
                "\t.reg .pred %p<2>;\n\t.reg .b64 x, y, z;\n\tsetp.eq.u32 %p1, %r1, 0;\n\tmov.u64 x, t;\n\tselp.b64 y, x, %rd2, %p1;\n\tmov.b64 z, y;\n\tst.global.u32 [z], %r1;",
            ),
            // Instructions of the shared state space: a table is not there, but the rule does not
            // follow state spaces, and an address stored as data can be loaded back by another thread.
            (
                "a matrix store through an address made from it",
                TABLE,
                "\tmov.u64 %rd1, t;\n\tstmatrix.sync.aligned.m8n8.x1.shared.b16 [%rd1], {%r1};",
            ),
            (
                "its address stored as data to shared memory",
                ".global .align 4 .b8 t[8] = {1, 0, 0, 0, 2, 0, 0, 0};\n.shared .align 8 .b8 s[8];",
                "\tmov.u64 %rd1, t;\n\tst.shared.u64 [s], %rd1;",
            ),
            // An address rebuilt from a comparison, or from the carry, without a value made from it.
            (
                "a store through an address chosen by a comparison of its address",
                TABLE,
                "\t.reg .pred %p<2>;\n\tmov.u64 %rd1, t;\n\tsetp.eq.u64 %p1, %rd1, 4096;\n\t@%p1 mov.u64 %rd2, 4096;\n\tst.global.u32 [%rd2], %r1;",
            ),
            (
                "a store through an address made from the carry of an addition to its address",
                TABLE,
                "\tmov.u64 %rd1, t;\n\tadd.cc.u64 %rd2, %rd1, 4;\n\taddc.u64 %rd3, 0, 0;\n\tst.global.u32 [%rd3], %r1;",
            ),
            // What only reads, but is not taught to the rule.
            (
                "a matrix load from an address made from it",
                TABLE,
                "\tmov.u64 %rd1, t;\n\tldmatrix.sync.aligned.m8n8.x1.shared.b16 {%r1}, [%rd1];",
            ),
            (
                "a test of the state space of its address",
                TABLE,
                "\t.reg .pred %p<2>;\n\tmov.u64 %rd1, t;\n\tisspacep.global %p1, %rd1;",
            ),
        ];
        let mut copied = Vec::new();
        for (what, declaration, other) in cases {
            let text = table_module(declaration, other);
            let info = ModuleInfo::analyse(text.as_bytes());
            if info.is_entry_independent("reader")
                || !info.copied_tables("reader").is_empty()
                || !info.dependent_vars("reader").contains(&"t".to_string())
            {
                copied.push(what);
            }
        }
        assert!(copied.is_empty(), "reader should be dependent with: {copied:#?}");
    }

    #[test]
    fn table_that_is_only_read_is_copied() {
        // What only reads `t`, by any way of writing it: `reader` and `other` keep copies.
        let cases = [
            ("a volatile load", "\tld.volatile.global.u32 %r1, [t];"),
            (
                "a load under a guard",
                "\t.reg .pred %p<2>;\n\tsetp.eq.u32 %p1, %r1, 0;\n\t@%p1 ld.global.u32 %r1, [t];",
            ),
            ("a vector load", "\tld.global.v2.u32 {%r1, %r2}, [t];"),
            (
                "a load through its generic address",
                "\tcvta.global.u64 %rd1, t;\n\tld.u32 %r1, [%rd1];",
            ),
            (
                "a load through registers whose names have no %",
                "\t.reg .pred %p<2>;\n\t.reg .b64 x, y;\n\tsetp.eq.u32 %p1, %r1, 0;\n\tmov.u64 x, t;\n\tselp.b64 y, x, %rd2, %p1;\n\tld.global.u32 %r2, [y+4];",
            ),
            ("a load written against its operands", "\tld.global.u32%r1, [t];"),
            (
                "its bytes stored elsewhere",
                "\tld.global.u32 %r1, [t+4];\n\tst.local.u32 [%rd3], %r1;\n\tst.global.u32 [%rd3], %r1;",
            ),
            (
                "a declaration of registers named as nothing else",
                "\t.reg .b64 p;\n\tmov.u64 p, t;\n\tld.global.u32 %r1, [p];",
            ),
        ];
        let mut dependent = Vec::new();
        for (what, other) in cases {
            let info = ModuleInfo::analyse(table_module(TABLE, other).as_bytes());
            if !info.is_entry_independent("reader") || info.copied_tables("other") != ["t"] {
                dependent.push(what);
            }
        }
        assert!(dependent.is_empty(), "t should be copied with: {dependent:#?}");
    }

    #[test]
    fn declaration_of_several_variables_is_not_copied() {
        // `u` is declared with `t`: the analysis knows `t` only, and `writer`, which writes `u`
        // without naming anything it knows, comes from the whole module after its sub-module
        // fails to compile. `reader` must then not hold a copy of `u` (2026-11-23: it did).
        let text = "\
.version 9.0
.target sm_70
.address_size 64

.global .align 4 .b8 t[4] = {1, 2, 3, 4}, u[4] = {5, 6, 7, 8};

.visible .entry reader(
\t.param .u64 reader_param_0
)
{
\t.reg .b32 %r<3>;
\t.reg .b64 %rd<3>;
\tld.param.u64 %rd1, [reader_param_0];
\tcvta.to.global.u64 %rd2, %rd1;
\tld.global.u32 %r1, [t];
\tld.global.u32 %r2, [u];
\tadd.u32 %r1, %r1, %r2;
\tst.global.u32 [%rd2], %r1;
\tret;
}

.visible .entry writer()
{
\tst.global.u32 [u], 9;
\tret;
}
";
        let info = ModuleInfo::analyse(text.as_bytes());
        assert!(!info.is_entry_independent("reader"));
        assert!(info.copied_tables("reader").is_empty());
    }

    #[test]
    fn function_not_analysed_disqualifies_every_table() {
        // A function whose body the analysis does not read could write `t`: then nothing is copied.
        // `w` is defined in a way the function scanner does not find (2026-11-23: `t` was copied,
        // and `other`, which calls `w`, came from the whole module once its sub-module failed to
        // compile), or with a body the statement scanner cannot read.
        let writers = [
            (
                "no space after .func",
                ".func(.reg .b32 rv) w(\n\t.reg .b32 a\n)\n{\n\tst.global.u32 [t], a;\n\tmov.u32 rv, 0;\n\tret;\n}",
            ),
            (
                "a comment after .func",
                ".func/* the writer */ (.reg .b32 rv) w(\n\t.reg .b32 a\n)\n{\n\tst.global.u32 [t], a;\n\tmov.u32 rv, 0;\n\tret;\n}",
            ),
            (
                "an unreadable body",
                ".func (.reg .b32 rv) w(\n\t.reg .b32 a\n)\n{\n\tst.global.u32 [t], a;\n\tmov.u32 rv, 0;\n\tret\n}",
            ),
        ];
        let mut copied = Vec::new();
        for (what, writer) in writers {
            let declaration = format!("{TABLE}\n{writer}");
            let text = table_module(&declaration, "\tcall (%r2), w, (%r1);");
            let info = ModuleInfo::analyse(text.as_bytes());
            if info.is_entry_independent("reader") || !info.copied_tables("reader").is_empty() {
                copied.push(what);
            }
        }
        assert!(copied.is_empty(), "reader should be dependent with: {copied:#?}");
    }

    #[test]
    fn table_address_passed_to_function_is_dependent() {
        // The address of a table is passed to a non-safe extern function: the entry is dependent
        // (the write analysis marks the variable as written through the call).
        let text = "\
.version 9.0
.target sm_70
.address_size 64

.global .align 4 .b8 tbl[16] = {0};
.extern .func process (.param .b64 ptr);

.visible .entry caller()
{
\t.reg .b64 %rd<2>;
\tmov.u64 %rd1, tbl;
\tcall.uni process, (%rd1);
\tret;
}

.visible .entry noop() { ret; }
";
        let info = ModuleInfo::analyse(text.as_bytes());
        assert!(
            !info.is_entry_independent("caller"),
            "caller passes tbl address to a non-safe call, should be dependent"
        );
    }
}
