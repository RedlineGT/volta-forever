//! A closed subset of scalar PTX. Registers receive a new identity at each declaration
//! (including named locals inside inline assembly), and labels resolve before evaluation.
use crate::scan::{self, Kind};
use std::collections::HashMap;

pub(super) type Register = usize;
#[derive(Clone, Debug)]
pub(super) enum Arg {
    Reg(Register),
    Imm(i64),
    Param(usize),
    Mem(Register, i64),
    Tuple(Vec<Arg>),
    Label(usize),
}
impl Arg {
    pub fn reg(&self) -> Option<Register> {
        if let Self::Reg(r) = self { Some(*r) } else { None }
    }
    pub fn visit(&self, f: &mut impl FnMut(Register)) {
        match self {
            Self::Reg(v) | Self::Mem(v, _) => f(*v),
            Self::Tuple(xs) => xs.iter().for_each(|x| x.visit(f)),
            _ => {}
        }
    }
}
#[derive(Debug)]
pub(super) struct Statement<'a> {
    pub op: &'a str,
    pub args: Vec<Arg>,
    pub guard: Option<(Register, bool)>,
    pub position: usize,
}
pub(super) struct Program<'a> {
    pub statements: Vec<Statement<'a>>,
    pub names: Vec<&'a str>,
    pub widths: Vec<u8>,
    pub roots: Vec<bool>,
    pub labels: HashMap<&'a str, usize>,
    pub params: Vec<&'a str>,
}
const TYPES: [&str; 12] = [
    ".u64", ".u32", ".u32", ".u64", ".u32", ".u32", ".u64", ".u64", ".u64", ".u64", ".u32", ".f32",
];

fn formal_names(signature: &str) -> Option<Vec<&str>> {
    let (args, shape) = signature.trim().strip_prefix('(')?.split_once(')')?;
    let shape = shape.split_whitespace().collect::<String>();
    if shape != ".maxntid128,1,1" {
        return None;
    }
    let mut names = Vec::new();
    for (index, field) in args.split(',').enumerate() {
        let mut fields = field.split_whitespace();
        if fields.next() != Some(".param") || fields.next() != TYPES.get(index).copied() {
            return None;
        }
        let name = scan::register(fields.next()?)?;
        if fields.next().is_some() || names.contains(&name) {
            return None;
        }
        names.push(name);
    }
    (names.len() == TYPES.len()).then_some(names)
}

fn integer(word: &str) -> Option<i64> {
    let bare = word.strip_suffix('U').unwrap_or(word);
    if let Some(x) = bare.strip_prefix("0f") {
        return (x.len() == 8).then(|| i64::from_str_radix(x, 16).ok()).flatten();
    }
    if let Some(x) = bare.strip_prefix("0x") {
        i64::from_str_radix(x, 16).ok()
    } else {
        bare.parse().ok()
    }
}
struct RegSet<'a> {
    // Families are declared once. Allocate identities lazily for members used in code.
    scopes: Vec<HashMap<&'a str, (usize, u8, bool)>>,
    ids: HashMap<(usize, &'a str), usize>,
    labels: HashMap<&'a str, usize>,
    names: Vec<&'a str>,
    widths: Vec<u8>,
    roots: Vec<bool>,
    serial: usize,
}
impl<'a> RegSet<'a> {
    fn declare(&mut self, operands: &'a str) -> Option<()> {
        let (ty, decls) = operands.trim().split_once(char::is_whitespace)?;
        let width = match ty {
            ".pred" => 1,
            ".b16" | ".u16" | ".f16" => 16,
            ".b32" | ".u32" | ".s32" | ".f32" => 32,
            ".b64" | ".u64" | ".s64" => 64,
            _ => return None,
        };
        for item in decls.split(',').map(str::trim) {
            let (name, len, family) = if let Some((s, n)) = item.split_once('<') {
                (s, n.strip_suffix('>')?.parse::<usize>().ok()?, true)
            } else {
                (item, 1, false)
            };
            if scan::register(name)? != name || len == 0 || len > 16384 {
                return None;
            }
            let scope = self.scopes.last_mut()?;
            if scope.insert(name, (len, width, family)).is_some() {
                return None;
            }
        }
        Some(())
    }
    fn reg(&mut self, name: &'a str) -> Option<Register> {
        let (scope, width) = if matches!(name, "%tid.x" | "%ctaid.x" | "%nctaid.x") {
            (0, 32)
        } else {
            scan::register(name)?;
            let digits = name.trim_end_matches(|ch: char| ch.is_ascii_digit());
            let suffix = &name[digits.len()..];
            if suffix.len() > 1 && suffix.starts_with('0') {
                return None;
            }
            let mut declared = None;
            for (depth, frame) in self.scopes.iter().enumerate().rev() {
                let direct = frame.get(name).filter(|d| !d.2);
                let family = frame
                    .get(digits)
                    .filter(|d| d.2 && suffix.parse::<usize>().is_ok_and(|i| i < d.0));
                if direct.is_some() && family.is_some() {
                    return None;
                }
                if let Some(d) = direct.or(family) {
                    declared = Some((if depth == 0 { 1 } else { self.serial + 1 }, d.1));
                    break;
                }
            }
            declared?
        };
        let next = self.names.len();
        if next >= 16384 {
            return None;
        }
        let id = *self.ids.entry((scope, name)).or_insert(next);
        if id == next {
            self.names.push(name);
            self.widths.push(width);
            self.roots.push(scope <= 1);
        }
        Some(id)
    }
    fn arg(&mut self, word: &'a str, params: &[&str]) -> Option<Arg> {
        let word = word.trim();
        if let Some(rest) = word.strip_prefix('[') {
            let address = rest.strip_suffix(']')?.trim();
            if let Some(i) = params.iter().position(|&name| name == address) {
                return Some(Arg::Param(i));
            }
            let (reg, off) = address.split_once('+').unwrap_or((address, "0"));
            let id = self.reg(reg.trim())?;
            return Some(Arg::Mem(id, integer(off.trim())?));
        }
        if let Some(inner) = word.strip_prefix('{') {
            return Some(Arg::Tuple(
                inner
                    .strip_suffix('}')?
                    .split(',')
                    .map(|v| self.arg(v, params))
                    .collect::<Option<_>>()?,
            ));
        }
        if let Some((r, p)) = word.split_once('|') {
            return Some(Arg::Tuple(vec![
                Arg::Reg(self.reg(r.trim())?),
                Arg::Reg(self.reg(p.trim())?),
            ]));
        }
        if let Some(x) = integer(word) {
            return Some(Arg::Imm(x));
        }
        Some(Arg::Reg(self.reg(word)?))
    }
}
impl<'a> Program<'a> {
    fn assigns(&self, pc: usize, reg: Register) -> bool {
        let stmt = &self.statements[pc];
        if stmt.op.starts_with("st.") || stmt.op.starts_with("bra") || stmt.op == "ret" {
            return false;
        }
        let mut changed = false;
        if let Some(dst) = stmt.args.first() {
            dst.visit(&mut |r| changed |= r == reg);
        }
        changed
    }
    pub(super) fn unchanged(&self, reg: Register, begin: usize, end: usize) -> bool {
        begin <= end && (begin..end).all(|pc| !self.assigns(pc, reg))
    }
    /// An observed value may guard the transformation only if its producing load
    /// dominates the insertion and no other instruction can assign that register.
    /// Walk both sides of conditional branches, including paths not taken in the
    /// one-row abstract execution.
    pub(super) fn immutable_guard(&self, reg: Register, origin: usize, target: usize, start: usize) -> bool {
        let source = &self.statements[origin];
        if source.op != "ld.global.nc.u32"
            || source.guard.is_some()
            || source.args.first().and_then(Arg::reg) != Some(reg)
        {
            return false;
        }
        if !self.unchanged(reg, 0, origin) || !self.unchanged(reg, origin + 1, self.statements.len()) {
            return false;
        }
        let mut pending = vec![start];
        let mut visited = vec![false; self.statements.len()];
        while let Some(pc) = pending.pop() {
            if pc == target {
                return false;
            }
            if pc == origin || pc >= visited.len() || visited[pc] {
                continue;
            }
            visited[pc] = true;
            let op = &self.statements[pc];
            if op.op == "ret" {
                continue;
            }
            if let [Arg::Label(to)] = op.args.as_slice() {
                pending.push(*to);
                if op.guard.is_none() {
                    continue;
                }
            }
            pending.push(pc + 1);
        }
        true
    }

    pub(super) fn parse(signature: &'a str, text: &'a str) -> Option<Self> {
        if text.len() > 1 << 20 || text.contains("%q4affine") || text.contains("$q4affine") {
            return None;
        }
        let params = formal_names(signature)?;
        let items = scan::items(text.as_bytes(), 0..text.len())?;
        if items.len() > 25000 {
            return None;
        }
        let mut regs = RegSet {
            scopes: vec![HashMap::new()],
            ids: HashMap::new(),
            labels: HashMap::new(),
            names: Vec::new(),
            widths: Vec::new(),
            roots: Vec::new(),
            serial: 0,
        };
        let mut pos = 0;
        let mut local_writes = std::collections::HashSet::new();
        for item in &items {
            match &item.kind {
                Kind::Label => {
                    let k = text[item.range.clone()].trim_end_matches(':').trim();
                    if regs.labels.insert(k, pos).is_some() {
                        return None;
                    }
                }
                Kind::Statement { opcode, .. } if &text[opcode.clone()] != ".reg" => pos += 1,
                _ => {}
            }
        }
        let mut statements = Vec::with_capacity(pos);
        for item in items {
            match &item.kind {
                Kind::Open => {
                    if regs.scopes.len() > 1 {
                        return None;
                    }
                    local_writes.clear();
                    regs.serial += 1;
                    regs.scopes.push(HashMap::new())
                }
                Kind::Close => {
                    if regs.scopes.len() < 2 {
                        return None;
                    }
                    regs.scopes.pop()?;
                }
                Kind::Line => {}
                Kind::Label if regs.scopes.len() == 1 => {}
                Kind::Label => return None,
                Kind::Statement {
                    guarded,
                    opcode,
                    operands,
                } => {
                    let op = &text[opcode.clone()];
                    let raw = &text[operands.clone()];
                    if op == ".reg" {
                        if *guarded {
                            return None;
                        }
                        regs.declare(raw)?;
                        continue;
                    }
                    let guard = if *guarded {
                        let guard = text[item.range.start..opcode.start].trim().strip_prefix('@')?;
                        let (reg, yes) = guard.strip_prefix('!').map_or((guard, true), |r| (r, false));
                        let r = regs.reg(reg)?;
                        if regs.widths[r] != 1 {
                            return None;
                        }
                        Some((r, yes))
                    } else {
                        None
                    };
                    let args = if matches!(op, "bra" | "bra.uni") {
                        vec![Arg::Label(*regs.labels.get(raw.trim())?)]
                    } else {
                        scan::split_operands(raw)
                            .iter()
                            .map(|a| regs.arg(a, &params))
                            .collect::<Option<_>>()?
                    };
                    if regs.scopes.len() > 1 {
                        if matches!(op, "bra" | "bra.uni" | "ret") {
                            return None;
                        }
                        if let Some((r, _)) = guard
                            && !regs.roots[r]
                            && !local_writes.contains(&r)
                        {
                            return None;
                        }
                        let (dst, src) = args.split_first()?;
                        let mut valid = true;
                        for a in src {
                            a.visit(&mut |r| {
                                if !regs.roots[r] && !local_writes.contains(&r) {
                                    valid = false;
                                }
                            });
                        }
                        if !valid {
                            return None;
                        }
                        if guard.is_none() {
                            dst.visit(&mut |r| {
                                if !regs.roots[r] {
                                    local_writes.insert(r);
                                }
                            });
                        }
                    }
                    statements.push(Statement {
                        op,
                        args,
                        guard,
                        position: item.range.start,
                    });
                }
            }
        }
        if regs.scopes.len() != 1 {
            return None;
        }
        Some(Self {
            statements,
            names: regs.names,
            widths: regs.widths,
            roots: regs.roots,
            labels: regs.labels,
            params,
        })
    }
}
