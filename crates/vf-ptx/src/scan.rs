//! The items of a function body (statements, labels, block braces) with their byte ranges, and the
//! pieces of a statement: guard, opcode, operands.

use std::ops::Range;

/// What an item of a function body is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// `{` that opens a nested block.
    Open,
    /// `}` that closes one.
    Close,
    /// `name:`
    Label,
    /// A debug directive without a semicolon terminator (`.loc`, `.file`).
    Line,
    /// An instruction or a directive ended by `;`.
    Statement {
        /// The statement starts with a guard predicate (`@%p` or `@!%p`).
        guarded: bool,
        opcode: Range<usize>,
        /// The text between the opcode and the `;`.
        operands: Range<usize>,
    },
}

/// An item and where its text is, terminator included.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    pub range: Range<usize>,
    pub kind: Kind,
}

/// Position after the comment that starts at `at`, if one does.
pub(crate) fn skip_comment(text: &[u8], at: usize) -> Option<usize> {
    match text.get(at..at + 2)? {
        b"//" => Some(
            text[at..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(text.len(), |p| at + p + 1),
        ),
        b"/*" => Some(
            text[at + 2..]
                .windows(2)
                .position(|w| w == b"*/")
                .map_or(text.len(), |p| at + 2 + p + 2),
        ),
        _ => None,
    }
}

/// Position after the string literal that starts at `at` (a `"`).
pub(crate) fn skip_string(text: &[u8], at: usize) -> usize {
    string_end(text, at).unwrap_or(text.len())
}

fn string_end(text: &[u8], at: usize) -> Option<usize> {
    let mut cursor = at + 1;
    while cursor < text.len() {
        match text[cursor] {
            b'\\' => cursor += 2,
            b'"' => return Some(cursor + 1),
            _ => cursor += 1,
        }
    }
    None
}

/// `text` with each comment and each string literal replaced by a space: what is left is what
/// the assembler reads names in.
pub(crate) fn without_comments(text: &[u8]) -> Vec<u8> {
    let mut code = Vec::with_capacity(text.len());
    let mut at = 0;
    while at < text.len() {
        let next = match skip_comment(text, at) {
            Some(next) => next,
            None if text[at] == b'"' => skip_string(text, at),
            None => {
                code.push(text[at]);
                at += 1;
                continue;
            }
        };
        code.push(b' ');
        at = next.min(text.len());
    }
    code
}

/// `text` with every comment written over with spaces, its ends of lines kept (a directive may
/// end with its line): a comment is not code, and what reads a statement as text would read it
/// with the rest. String literals stay as they are.
pub(crate) fn blank_comments(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut code = Vec::with_capacity(bytes.len());
    let mut at = 0;
    while at < bytes.len() {
        if let Some(next) = skip_comment(bytes, at) {
            let next = next.min(bytes.len());
            code.extend(
                bytes[at..next]
                    .iter()
                    .map(|&byte| if byte == b'\n' { b'\n' } else { b' ' }),
            );
            at = next;
        } else if bytes[at] == b'"' {
            let next = skip_string(bytes, at).min(bytes.len());
            code.extend_from_slice(&bytes[at..next]);
            at = next;
        } else {
            code.push(bytes[at]);
            at += 1;
        }
    }
    // (A comment starts and ends with ASCII, and so does a string: what is copied is whole
    // characters, and what is written over becomes spaces.)
    String::from_utf8(code).unwrap_or_else(|error| String::from_utf8_lossy(error.as_bytes()).into_owned())
}

fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$' | b'%')
}

/// Position of the `;` that ends the statement starting at `at`, comments and strings skipped.
fn statement_end(text: &[u8], at: usize, limit: usize) -> Option<usize> {
    let mut cursor = at;
    while cursor < limit {
        if let Some(next) = skip_comment(text, cursor) {
            cursor = next;
        } else if text[cursor] == b'"' {
            cursor = skip_string(text, cursor);
        } else if text[cursor] == b';' {
            return Some(cursor);
        } else {
            cursor += 1;
        }
    }
    None
}

/// Skip whitespace and comments between the fields of a debug directive.
fn debug_space(mut text: &[u8]) -> &[u8] {
    loop {
        text = text.trim_ascii_start();
        match skip_comment(text, 0) {
            Some(end) => text = &text[end..],
            None => return text,
        }
    }
}

fn debug_keyword<'a>(text: &'a [u8], keyword: &[u8]) -> Option<&'a [u8]> {
    let tail = text.strip_prefix(keyword)?;
    (tail.first().is_some_and(u8::is_ascii_whitespace) || skip_comment(tail, 0).is_some()).then_some(tail)
}

fn debug_number(text: &[u8]) -> Option<&[u8]> {
    let text = debug_space(text);
    let end = text.iter().take_while(|b| b.is_ascii_alphanumeric()).count();
    let word = text[..end].strip_suffix(b"U").unwrap_or(&text[..end]);
    let valid = if let Some(hex) = word.strip_prefix(b"0x").or_else(|| word.strip_prefix(b"0X")) {
        !hex.is_empty() && hex.iter().all(u8::is_ascii_hexdigit)
    } else if let Some(binary) = word.strip_prefix(b"0b").or_else(|| word.strip_prefix(b"0B")) {
        !binary.is_empty() && binary.iter().all(|b| matches!(b, b'0' | b'1'))
    } else {
        !word.is_empty() && word.iter().all(u8::is_ascii_digit)
    };
    valid.then_some(&text[end..])
}

fn debug_position(text: &[u8]) -> Option<&[u8]> {
    debug_number(debug_number(debug_number(text)?)?)
}

/// Each directive ends after its last operand. The next label or instruction can be on
/// the very same line; swallowing it would hide register writes from the lowering passes.
fn debug_end(text: &[u8], directive: &[u8]) -> Option<usize> {
    let args = &text[directive.len()..];
    let mut tail;
    if directive == b".file" {
        let filename = debug_space(debug_number(args)?);
        filename.strip_prefix(b"\"")?;
        tail = &filename[string_end(filename, 0)?..];
        if let Some(timestamp) = debug_space(tail).strip_prefix(b",") {
            let size = debug_space(debug_number(timestamp)?).strip_prefix(b",")?;
            tail = debug_number(size)?;
        }
    } else {
        tail = debug_position(args)?;
        if directive == b".loc"
            && let Some(clause) = debug_space(tail).strip_prefix(b",")
        {
            let name = debug_space(debug_keyword(debug_space(clause), b"function_name")?);
            let end = name.iter().take_while(|&&b| is_identifier_byte(b) || b == b'.').count();
            if end == 0 {
                return None;
            }
            tail = &name[end..];
            if let Some(offset) = debug_space(tail).strip_prefix(b"+") {
                tail = debug_number(offset)?;
            }
            if let Some(clause) = debug_space(tail).strip_prefix(b",") {
                tail = debug_position(debug_keyword(debug_space(clause), b"inlined_at")?)?;
            }
        }
    }
    Some(text.len() - tail.len())
}

/// The items of the function body `text[body]` (`body` includes the outer braces, which are not
/// returned). `None` when the text does not divide into items: an unterminated statement or
/// unbalanced braces.
pub fn items(text: &[u8], body: Range<usize>) -> Option<Vec<Item>> {
    let limit = body.end.checked_sub(1)?;
    if text.get(body.start) != Some(&b'{') || text.get(limit) != Some(&b'}') {
        return None;
    }
    let mut found = Vec::new();
    let mut depth = 0usize;
    let mut at = body.start + 1;
    while at < limit {
        if let Some(next) = skip_comment(text, at) {
            at = next.min(limit);
            continue;
        }
        let byte = text[at];
        if byte.is_ascii_whitespace() {
            at += 1;
            continue;
        }
        if byte == b'{' || byte == b'}' {
            let kind = if byte == b'{' {
                depth += 1;
                Kind::Open
            } else {
                depth = depth.checked_sub(1)?;
                Kind::Close
            };
            found.push(Item {
                range: at..at + 1,
                kind,
            });
            at += 1;
            continue;
        }
        let rest = &text[at..limit];
        if let Some(directive) = [b".loc".as_slice(), b".loc_intermediate", b".file"]
            .into_iter()
            .find(|d| debug_keyword(rest, d).is_some())
        {
            let end = at + debug_end(rest, directive)?;
            found.push(Item {
                range: at..end,
                kind: Kind::Line,
            });
            at = end;
            continue;
        }
        // A label: an identifier followed by a single `:`.
        let word = text[at..limit].iter().take_while(|&&b| is_identifier_byte(b)).count();
        if word > 0 {
            let after = at
                + word
                + text[at + word..limit]
                    .iter()
                    .take_while(|b| b.is_ascii_whitespace())
                    .count();
            if text.get(after) == Some(&b':') && text.get(after + 1) != Some(&b':') && after < limit {
                found.push(Item {
                    range: at..after + 1,
                    kind: Kind::Label,
                });
                at = after + 1;
                continue;
            }
        }
        let end = statement_end(text, at, limit)?;
        let mut cursor = at;
        let guarded = text[cursor] == b'@';
        if guarded {
            cursor += text[cursor..end]
                .iter()
                .take_while(|b| !b.is_ascii_whitespace())
                .count();
            cursor += text[cursor..end].iter().take_while(|b| b.is_ascii_whitespace()).count();
        }
        let opcode_end = cursor
            + text[cursor..end]
                .iter()
                .take_while(|b| !b.is_ascii_whitespace())
                .count();
        found.push(Item {
            range: at..end + 1,
            kind: Kind::Statement {
                guarded,
                opcode: cursor..opcode_end,
                operands: opcode_end..end,
            },
        });
        at = end + 1;
    }
    (depth == 0).then_some(found)
}

/// The operands of a statement: `operands` split at the commas that are outside braces, brackets and
/// parentheses, each trimmed.
pub fn split_operands(operands: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (at, byte) in operands.bytes().enumerate() {
        match byte {
            b'{' | b'[' | b'(' => depth += 1,
            b'}' | b']' | b')' => depth = depth.saturating_sub(1),
            b',' if depth == 0 => {
                parts.push(operands[start..at].trim());
                start = at + 1;
            }
            _ => {}
        }
    }
    let last = operands[start..].trim();
    if !last.is_empty() || !parts.is_empty() {
        parts.push(last);
    }
    parts
}

/// The names in `text` that can be registers, in order: `%r1`, and also bare identifiers, which inline
/// assembly may declare as registers (`.reg .f16 low, high;`). Numbers are skipped. A special register
/// with a component (`%tid.x`) comes out as two names; neither is ever written, which is all the
/// callers care about.
pub fn names(text: &str) -> impl Iterator<Item = &str> {
    let bytes = text.as_bytes();
    let is_name_byte = |byte: u8| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$');
    let mut at = 0;
    std::iter::from_fn(move || {
        while at < bytes.len() {
            let start = at;
            if bytes[at].is_ascii_digit() {
                // A number, with its base or type letters (`0f3F800000`, `0x1F`, `16U`).
                while at < bytes.len() && is_name_byte(bytes[at]) {
                    at += 1;
                }
            } else if bytes[at] == b'%' || is_name_byte(bytes[at]) {
                at += 1;
                while at < bytes.len() && is_name_byte(bytes[at]) {
                    at += 1;
                }
                if at > start + 1 || bytes[start] != b'%' {
                    return Some(&text[start..at]);
                }
            } else {
                at += 1;
            }
        }
        None
    })
}

/// The registers of a brace-enclosed vector operand `{%a, %b}`; `None` if the operand is not one or
/// holds something else than registers.
pub fn vector(operand: &str) -> Option<Vec<&str>> {
    vector_with(operand, |item| register(item).is_some())
}

/// The members of a brace-enclosed operand that all satisfy a particular operand form.
pub fn vector_with(operand: &str, accepts: impl Fn(&str) -> bool) -> Option<Vec<&str>> {
    let inner = operand.strip_prefix('{')?.strip_suffix('}')?;
    inner
        .split(',')
        .map(|part| part.trim())
        .map(|item| accepts(item).then_some(item))
        .collect()
}

/// `operand` if it is a single register name.
pub fn register(operand: &str) -> Option<&str> {
    let mut found = names(operand);
    match (found.next(), found.next()) {
        (Some(name), None) if name.len() == operand.len() => Some(name),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comments_are_written_over_and_strings_kept() {
        let text = "a /* b\nc */ d // e\n.file 1 \"x//y/*z\" f";
        let blanked = blank_comments(text);
        assert_eq!(blanked.len(), text.len());
        let ends = |text: &str| text.match_indices('\n').map(|(at, _)| at).collect::<Vec<_>>();
        assert_eq!(ends(&blanked), ends(text));
        assert_eq!(
            blanked.split_whitespace().collect::<Vec<_>>(),
            ["a", "d", ".file", "1", "\"x//y/*z\"", "f"]
        );
        // A comment that does not end is one to the end of the text.
        assert_eq!(blank_comments("a /* b"), "a     ");
        assert_eq!(blank_comments("é // è\né"), "é      \né");
    }

    fn kinds(text: &str) -> Vec<(String, &'static str)> {
        items(text.as_bytes(), 0..text.len())
            .unwrap()
            .into_iter()
            .map(|item| {
                let name = match item.kind {
                    Kind::Open => "open",
                    Kind::Close => "close",
                    Kind::Label => "label",
                    Kind::Line => "line",
                    Kind::Statement { guarded: true, .. } => "guarded",
                    Kind::Statement { .. } => "statement",
                };
                (text[item.range].to_string(), name)
            })
            .collect()
    }

    #[test]
    fn divides_a_body_into_items() {
        let text = "{\n\t.reg .b32 %r<4>;\n\t.loc 1 23 0, function_name $L__info_string0, inlined_at 1 5 0\n\
                    $L__BB0_1: // loop; {\n\tadd.s32 %r1, %r1, 1;\n\t@%p1 bra $L__BB0_1;\n\
                    \t{ .reg .b64 %tmp;\n\tcvta.shared.u64 %rd1, %tmp; }\n\t.pragma \"a;b\";\n\tret;\n}";
        let found = kinds(text);
        let expected = [
            (".reg .b32 %r<4>;", "statement"),
            (".loc 1 23 0, function_name $L__info_string0, inlined_at 1 5 0", "line"),
            ("$L__BB0_1:", "label"),
            ("add.s32 %r1, %r1, 1;", "statement"),
            ("@%p1 bra $L__BB0_1;", "guarded"),
            ("{", "open"),
            (".reg .b64 %tmp;", "statement"),
            ("cvta.shared.u64 %rd1, %tmp;", "statement"),
            ("}", "close"),
            (".pragma \"a;b\";", "statement"),
            ("ret;", "statement"),
        ];
        assert_eq!(found.len(), expected.len());
        for (found, expected) in found.iter().zip(expected) {
            assert_eq!((found.0.as_str(), found.1), expected);
        }
    }

    #[test]
    fn debug_directives_stop_before_same_line_code() {
        for location in [
            ".loc 1 2 3",
            ".loc 0x1 02 0b1U",
            ".loc_intermediate 4 5 6",
            ".loc 1 2 3, function_name f",
            ".loc 1 2 3, function_name .debug_str + 16, inlined_at 3 4 5",
            ".loc 1/* comment */2 3",
            r##".file 1 "a file; with \"quotes\"""##,
            r#".file 1 "k" , 1700, 901"#,
        ] {
            let text = format!("{{ {location} L: @%p mov.u32 %r,0; }}");
            assert_eq!(
                kinds(&text),
                [
                    (location.to_owned(), "line"),
                    ("L:".to_owned(), "label"),
                    ("@%p mov.u32 %r,0;".to_owned(), "guarded"),
                ]
            );
        }
        for bad in [
            ".loc 1 2",
            ".loc 1 2 0, xyz",
            ".loc 1 2 0, function_name label+, inlined_at 1 2 3",
        ] {
            let text = format!("{{ {bad} mov.u32 %r,7; }}");
            assert!(items(text.as_bytes(), 0..text.len()).is_none(), "{bad}");
        }
    }

    #[test]
    fn finds_opcode_and_operands() {
        let text = "{ @!%p3 mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%r1, %r2}, {%r3}, {%r4} ; ret; }";
        let found = items(text.as_bytes(), 0..text.len()).unwrap();
        let Kind::Statement {
            guarded,
            opcode,
            operands,
        } = &found[0].kind
        else {
            panic!("not a statement");
        };
        assert!(*guarded);
        assert_eq!(&text[opcode.clone()], "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32");
        assert_eq!(
            split_operands(&text[operands.clone()]),
            ["{%r1, %r2}", "{%r3}", "{%r4}"]
        );
        let Kind::Statement { opcode, operands, .. } = &found[1].kind else {
            panic!("not a statement");
        };
        assert_eq!(&text[opcode.clone()], "ret");
        assert!(split_operands(&text[operands.clone()]).is_empty());
    }

    #[test]
    fn a_state_space_qualifier_is_not_a_label() {
        let text = "{\n\tldmatrix.sync.aligned.m8n8.x1.shared::cta.b16 {%r1}, [%r2];\n}";
        assert_eq!(kinds(text).len(), 1);
    }

    #[test]
    fn rejects_what_does_not_divide() {
        assert!(items(b"{ add.s32 %r1, %r1, 1 }", 0..23).is_none());
        assert!(items(b"{ } }", 0..5).is_none());
        assert!(items(b"{ { ret; }", 0..10).is_none());
    }

    #[test]
    fn splits_operands_and_finds_registers() {
        assert_eq!(
            split_operands(" {%r1, %r2}, [%rd3+16], (%f1, %f2), 0f3F800000 "),
            ["{%r1, %r2}", "[%rd3+16]", "(%f1, %f2)", "0f3F800000"]
        );
        assert_eq!(
            names("{%r1, low}|%p1, [%rd3+16], %tid.x, 0f3F800000, 0x1F").collect::<Vec<_>>(),
            ["%r1", "low", "%p1", "%rd3", "%tid", "x"]
        );
        assert_eq!(vector("{%r1, %r2}"), Some(vec!["%r1", "%r2"]));
        assert_eq!(vector("{%r1, 5}"), None);
        assert_eq!(vector("%r1"), None);
        assert_eq!(register("%rd85"), Some("%rd85"));
        assert_eq!(register("high"), Some("high"));
        assert_eq!(register("%rd85+4"), None);
        assert_eq!(register("16"), None);
    }
}
