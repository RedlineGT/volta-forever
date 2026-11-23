//! Pairs of FP16 products in program text (`docs/pairs.md`, section 13): two products with f32
//! accumulators and the same A, both operands from plain `ldmatrix`, are computed together on two
//! of Volta's `m8n8k4` per `m16n8k8` (each alone takes four, half of whose results are dropped),
//! with their accumulators kept in the program's own registers, permuted between lanes.
//!
//! What it rests on, each part proven here before anything is paired:
//!
//! - **The same bits.** A product by itself whose operands come from plain `ldmatrix`
//!   (`Plan::in_halves`) is two of Volta's products per `m16n8k8`, k 0..3 then k 4..7, from the
//!   accumulator in place. A pair does those same instructions on the same rows of A and columns of
//!   B, each term of k in the same slot, in the same order, for both products at once: every result
//!   is the same sum of the same terms. Products added otherwise today (B from `ldmatrix.trans`,
//!   operands gathered from other lanes, chains, f16 accumulators) are not paired.
//! - **One state, everywhere.** In the arrangement (`docs/pairs.md`, section 2) register i of lane L
//!   holds register `i & 3` of the accumulator of x (i < 4) or y as Turing's lane `pi(L)` holds it.
//!   The registers of pairs (the set S) are written by nothing but the products of pairs and by
//!   unguarded copies (`mov`) of a literal, of a register of S, or of a register that holds one
//!   literal in every lane. Each of these keeps "lane L holds what lane `pi(L)` holds in Turing's
//!   layout" true of what it writes (a literal is the same in both layouts, a copy between two
//!   registers so permuted is a copy of the permuted values): so it is true of every register of S
//!   at every place, and nothing has to follow where values flow.
//! - **Conversions.** Any other statement that reads a register of S reads it in Turing's layout:
//!   one shuffle in place before the statement (index `pi^-1(L)`), and one after it back (index
//!   `pi(L)`) where the register may still be read on some way (liveness; where it may not, it is
//!   left in Turing's layout, and nothing reads it before it is written again). A shuffle needs the 32
//!   lanes: it is put only where, whenever a lane runs that place, all 32 run it together
//!   (`whole_places`). A reader anywhere else drops the pairs of the registers it reads.
//!
//! Anything the pass cannot show leaves the products to `Body::mma_f32`, as they were.

use std::collections::{HashMap, HashSet};

use crate::scan::{self, Item, Kind};

use super::parse::{Tensor, parse};
use super::plan::Plan;
use super::types::{first_operand, integer};

/// The two products of a pair, computed at the second one.
#[derive(Debug)]
pub(super) struct Pair<'a> {
    /// The four accumulator registers of the first product (x) and of the second (y).
    pub(super) x: Vec<&'a str>,
    pub(super) y: Vec<&'a str>,
    /// Their A registers, two for each `m16n8k8`.
    pub(super) a: Vec<&'a str>,
    /// Their B registers, one for each `m16n8k8`.
    pub(super) bx: Vec<&'a str>,
    pub(super) by: Vec<&'a str>,
}

/// What the plan of a function body says of pairs. Item indices, as `Body::current_index`.
#[derive(Debug, Default)]
pub(super) struct Pairs<'a> {
    /// The second product of each pair: both are computed there.
    pub(super) second: HashMap<usize, Pair<'a>>,
    /// The first product of each pair: nothing is computed there.
    pub(super) first: HashSet<usize>,
    /// A operands (the two registers of an `m16n8k8`) that their `ldmatrix` loads as the rows of
    /// pairs (`Form::QRows`), in the order of the text.
    pub(super) rows: Vec<(&'a str, &'a str)>,
    /// B operands (x's register, y's register) that their `ldmatrix` loads as the columns of pairs
    /// (`Form::QCols`): x's where `h` is clear, y's where it is set.
    pub(super) cols: Vec<(&'a str, &'a str)>,
    /// Registers of pairs that a statement reads: put in Turing's layout before it.
    pub(super) before: HashMap<usize, Vec<&'a str>>,
    /// Those of them that may be read again: put back after it.
    pub(super) after: HashMap<usize, Vec<&'a str>>,
}

/// How control leaves an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    /// To the next instruction.
    Next,
    /// To the next instruction, after a function that may do anything with the lanes.
    Call,
    /// `bra` under no predicate.
    Jump,
    /// `bra` under a predicate: to the label or to the next instruction.
    Branch,
    /// `ret`, `exit` or `trap` under no predicate.
    Leave,
    /// The same under a predicate: some lanes may leave.
    MayLeave,
}

struct Instruction<'a> {
    item: usize,
    guarded: bool,
    opcode: &'a str,
    operands: &'a str,
    flow: Flow,
    target: Option<&'a str>,
    /// A `{` or `}` stands between this instruction and the one before.
    scope: bool,
}

/// A predecessor that is no instruction: the entry of the function.
const ENTRY: usize = usize::MAX;

/// The instructions of the body and its control-flow graph: `None` when it cannot be known (an
/// indirect branch, a label defined twice or named and not defined).
struct Graph<'a> {
    instructions: Vec<Instruction<'a>>,
    /// Successors of each instruction; `n` (the number of instructions) is the end of the function.
    successors: Vec<Vec<usize>>,
    /// Predecessors of each instruction, `ENTRY` for the first.
    predecessors: Vec<Vec<usize>>,
}

impl<'a> Graph<'a> {
    fn new(text: &'a str, items: &[Item]) -> Option<Graph<'a>> {
        let mut instructions: Vec<Instruction> = Vec::new();
        let mut labels: HashMap<&str, usize> = HashMap::new();
        let mut pending: Vec<&str> = Vec::new();
        let mut scope = false;
        for (index, item) in items.iter().enumerate() {
            match &item.kind {
                Kind::Label => {
                    let name = scan::names(&text[item.range.clone()]).next()?;
                    if labels.contains_key(name) || pending.contains(&name) {
                        return None;
                    }
                    pending.push(name);
                }
                Kind::Open | Kind::Close => scope = true,
                Kind::Line => {}
                Kind::Statement {
                    guarded,
                    opcode,
                    operands,
                } => {
                    let opcode = &text[opcode.clone()];
                    if opcode.starts_with('.') {
                        continue;
                    }
                    let operands = &text[operands.clone()];
                    let (flow, target) = match opcode.split('.').next() {
                        Some("bra") => (
                            if *guarded { Flow::Branch } else { Flow::Jump },
                            Some(scan::register(operands.trim())?),
                        ),
                        Some("brx") => return None,
                        Some("call") => (Flow::Call, None),
                        Some("ret" | "exit" | "trap") => (if *guarded { Flow::MayLeave } else { Flow::Leave }, None),
                        _ => (Flow::Next, None),
                    };
                    for label in pending.drain(..) {
                        labels.insert(label, instructions.len());
                    }
                    instructions.push(Instruction {
                        item: index,
                        guarded: *guarded,
                        opcode,
                        operands,
                        flow,
                        target,
                        scope,
                    });
                    scope = false;
                }
            }
        }
        let n = instructions.len();
        for label in pending {
            labels.insert(label, n);
        }
        let mut successors = Vec::with_capacity(n);
        for (k, instruction) in instructions.iter().enumerate() {
            let target = || instruction.target.and_then(|label| labels.get(label).copied());
            successors.push(match instruction.flow {
                Flow::Next | Flow::Call => vec![k + 1],
                Flow::Jump => vec![target()?],
                Flow::Branch => vec![k + 1, target()?],
                Flow::Leave => vec![n],
                Flow::MayLeave => vec![k + 1, n],
            });
        }
        let mut predecessors = vec![Vec::new(); n];
        if n > 0 {
            predecessors[0].push(ENTRY);
        }
        for (k, next) in successors.iter().enumerate() {
            for &s in next {
                if s < n {
                    predecessors[s].push(k);
                }
            }
        }
        Some(Graph {
            instructions,
            successors,
            predecessors,
        })
    }

    /// Control goes from instruction k to one place, with every lane that runs it.
    fn single(&self, k: usize) -> bool {
        matches!(self.instructions[k].flow, Flow::Next | Flow::Jump)
    }
}

/// An instruction that the 32 lanes of the warp run together, by the PTX manual: `.aligned` "all
/// threads in the warp must execute the same instruction", and in conditional code only where the
/// condition is the same in every lane.
fn aligned(instruction: &Instruction) -> bool {
    !instruction.guarded
        && instruction.opcode.contains(".sync.aligned")
        && matches!(
            instruction.opcode.split('.').next(),
            Some("mma" | "ldmatrix" | "stmatrix" | "movmatrix" | "wmma")
        )
}

/// For each instruction, the nearest aligned instruction that post-dominates it: one that every path
/// from it to the end of the function passes (a call may not come back to the caller: it counts as
/// a way to the end). `None` where there is none, and where the end cannot be reached at all.
/// Immediate post-dominators by Cooper, Harvey and Kennedy's iteration on the reversed graph.
fn aligned_post_dominators(graph: &Graph) -> Vec<Option<usize>> {
    let n = graph.instructions.len();
    let successors = |k: usize| {
        let mut next = graph.successors[k].clone();
        if graph.instructions[k].flow == Flow::Call {
            next.push(n);
        }
        next
    };
    // The reversed graph from the end: its successors are the predecessors in the graph.
    let mut reversed: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for k in 0..n {
        for s in successors(k) {
            reversed[s].push(k);
        }
    }
    let mut number = vec![usize::MAX; n + 1];
    let mut postorder: Vec<usize> = Vec::with_capacity(n + 1);
    let mut visited = vec![false; n + 1];
    let mut stack: Vec<(usize, usize)> = vec![(n, 0)];
    visited[n] = true;
    while let Some(&mut (node, ref mut next)) = stack.last_mut() {
        if let Some(&child) = reversed[node].get(*next) {
            *next += 1;
            if !visited[child] {
                visited[child] = true;
                stack.push((child, 0));
            }
        } else {
            number[node] = postorder.len();
            postorder.push(node);
            stack.pop();
        }
    }
    let mut parent = vec![usize::MAX; n + 1];
    parent[n] = n;
    let mut changed = true;
    while changed {
        changed = false;
        for &node in postorder.iter().rev().skip(1) {
            let mut found = usize::MAX;
            for s in successors(node) {
                if parent[s] == usize::MAX {
                    continue;
                }
                found = if found == usize::MAX {
                    s
                } else {
                    let (mut a, mut b) = (found, s);
                    while a != b {
                        while number[a] < number[b] {
                            a = parent[a];
                        }
                        while number[b] < number[a] {
                            b = parent[b];
                        }
                    }
                    a
                };
            }
            if found != parent[node] {
                parent[node] = found;
                changed = true;
            }
        }
    }
    // (A post-dominator comes before what it post-dominates in this order.)
    let mut nearest: Vec<Option<usize>> = vec![None; n + 1];
    for &node in postorder.iter().rev().skip(1) {
        nearest[node] = if aligned(&graph.instructions[node]) {
            Some(node)
        } else if parent[node] < n {
            nearest[parent[node]]
        } else {
            None
        };
    }
    nearest.truncate(n);
    nearest
}

/// The instructions that are whole places: whenever one lane runs one, the 32 lanes of the warp run
/// it together. The rules, `docs/pairs.md` section 7 for program text, applied until nothing changes:
///
/// - an aligned instruction is whole;
/// - **lead**: an instruction from which control goes to one place only, entered from nowhere
///   else, that is whole, is whole (every lane there goes there, and nobody joins it);
/// - **join**: an instruction is whole when every way into it comes from a whole instruction and is
///   taken by all its lanes or by none: control goes on to one place, or the way is one of a
///   uniform branch;
/// - **uniform**: a branch under a predicate at a whole place is taken by all lanes or by none when
///   an aligned instruction I depends on it: I post-dominates one way (every path from there to the
///   end passes I) and the other way can reach the end without I. The manual asks that the lanes
///   evaluate the condition of such a branch alike, and all 32 are there.
///
/// The entry of the function is not taken to be whole (a function may be called by some lanes).
fn whole_places(graph: &Graph) -> Vec<bool> {
    let n = graph.instructions.len();
    let lead = aligned_post_dominators(graph);
    // For an aligned instruction I: the instructions from which the end is reached without I.
    let mut escapes: HashMap<usize, Vec<bool>> = HashMap::new();
    let mut escape = |from: usize, avoided: usize| -> bool {
        if from == n {
            return true;
        }
        let reach = escapes.entry(avoided).or_insert_with(|| {
            let mut reach = vec![false; n];
            let mut stack: Vec<usize> = (0..n)
                .filter(|&k| k != avoided && graph.successors[k].contains(&n))
                .collect();
            for &k in &stack {
                reach[k] = true;
            }
            while let Some(k) = stack.pop() {
                for &p in &graph.predecessors[k] {
                    if p != ENTRY && p != avoided && !reach[p] {
                        reach[p] = true;
                        stack.push(p);
                    }
                }
            }
            reach
        });
        reach[from]
    };
    let mut uniform: Vec<Option<bool>> = vec![None; n];
    let mut whole: Vec<bool> = graph.instructions.iter().map(aligned).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for k in 0..n {
            if whole[k] || graph.predecessors[k].is_empty() {
                continue;
            }
            let mut all = true;
            for &p in &graph.predecessors[k] {
                let ok = p != ENTRY
                    && whole[p]
                    && (graph.single(p)
                        || (graph.instructions[p].flow == Flow::Branch
                            && *uniform[p].get_or_insert_with(|| {
                                let (next, taken) = (graph.successors[p][0], graph.successors[p][1]);
                                let reaches = |s: usize| if s < n { lead[s] } else { None };
                                reaches(taken).is_some_and(|i| escape(next, i))
                                    || reaches(next).is_some_and(|i| escape(taken, i))
                            })));
                if !ok {
                    all = false;
                    break;
                }
            }
            if all {
                whole[k] = true;
                changed = true;
            }
        }
        for k in (0..n).rev() {
            if !whole[k] && graph.single(k) {
                let next = graph.successors[k][0];
                if next < n && whole[next] && graph.predecessors[next] == [k] {
                    whole[k] = true;
                    changed = true;
                }
            }
        }
    }
    whole
}

/// A product that may be paired: FP16 with f32 accumulators, in place, not under a predicate,
/// computed by itself today in the halves of k, not a product of a chain.
struct Candidate<'a> {
    d: Vec<&'a str>,
    a: Vec<&'a str>,
    b: Vec<&'a str>,
}

fn candidate<'a>(instruction: &Instruction<'a>, plan: &Plan<'a>) -> Option<Candidate<'a>> {
    if instruction.guarded || !instruction.opcode.starts_with("mma.") || plan.chain_products.contains(&instruction.item)
    {
        return None;
    }
    let Ok(Tensor::Product {
        half: false,
        d,
        a,
        b,
        c,
    }) = parse(instruction.opcode, instruction.operands, &plan.constants16)
    else {
        return None;
    };
    let distinct = d.iter().collect::<HashSet<_>>().len() == d.len();
    let halves = a.len() == 2 * b.len() && a.chunks(2).zip(&b).all(|(a, b)| plan.in_halves(a, b));
    (d == c && distinct && halves).then_some(Candidate { d, a, b })
}

/// A literal that a `mov` can write: the same value in every lane.
pub(super) fn literal(operand: &str) -> bool {
    let digits = operand.strip_prefix('-').unwrap_or(operand);
    let float = |prefix: &str| {
        digits
            .strip_prefix(prefix)
            .is_some_and(|hex| hex.len() == 8 && hex.bytes().all(|b| b.is_ascii_hexdigit()))
    };
    integer(digits).is_some() || float("0f") || float("0F")
}

/// The register an unguarded `mov` of 32 bits writes into S, when what it copies keeps the state of
/// S: a literal, a register of S, or a register that holds one literal in every lane (`uniform`).
/// Also the register of S it copies, if any.
fn copy_into<'a>(
    instruction: &Instruction<'a>,
    registers: &HashSet<&'a str>,
    uniform: &HashSet<&str>,
) -> Option<(&'a str, Option<&'a str>)> {
    if instruction.guarded || !matches!(instruction.opcode, "mov.b32" | "mov.u32" | "mov.s32" | "mov.f32") {
        return None;
    }
    let [destination, source] = scan::split_operands(instruction.operands)[..] else {
        return None;
    };
    let destination = scan::register(destination).filter(|name| registers.contains(name))?;
    match scan::register(source) {
        Some(source) if registers.contains(source) => Some((destination, Some(source))),
        Some(source) if uniform.contains(source) => Some((destination, None)),
        Some(_) => None,
        None => literal(source).then_some((destination, None)),
    }
}

/// The registers of S that the instruction reads, when it does not write any: what follows its
/// first operand (the first operand of an instruction that writes no register, such as the address
/// of a store, counts as written: such an instruction drops the pairs).
fn read<'a>(instruction: &Instruction<'a>, registers: &HashSet<&'a str>) -> Vec<&'a str> {
    let first = first_operand(instruction.operands);
    let mut found: Vec<&str> = Vec::new();
    for name in scan::names(&instruction.operands[first.len()..]) {
        if registers.contains(name) && !found.contains(&name) {
            found.push(name);
        }
    }
    found
}

/// The pairs of a function body, from the plan of its products and operands. `clean`: the register
/// is declared in the body with 32 bits and not declared again in a block. `uniform`: registers that
/// hold one literal wherever they are read (written once, by an unguarded `mov` of it).
pub(super) fn plan<'a>(
    text: &'a str,
    items: &[Item],
    plan: &Plan<'a>,
    clean: &dyn Fn(&str) -> bool,
    uniform: &HashSet<&str>,
) -> Pairs<'a> {
    let Some(graph) = Graph::new(text, items) else {
        return Pairs::default();
    };
    let n = graph.instructions.len();
    let candidates: Vec<Option<Candidate>> = graph.instructions.iter().map(|i| candidate(i, plan)).collect();
    if candidates.iter().flatten().count() < 2 {
        return Pairs::default();
    }
    // Two products with the same A, the first computed at the second: straight-line code between
    // them (entered from nowhere else, no bound of a block), nothing between that names the first's
    // accumulator or loads its operands again.
    let mut partner: Vec<Option<usize>> = vec![None; n];
    let mut found: Vec<(usize, usize)> = Vec::new();
    for k in 0..n {
        let Some(x) = &candidates[k] else { continue };
        if partner[k].is_some() {
            continue;
        }
        let loads: HashSet<usize> =
            x.a.iter()
                .chain(&x.b)
                .filter_map(|register| plan.origins.get(register))
                .map(|origin| origin.statement)
                .collect();
        for j in k + 1..n {
            if graph.predecessors[j] != [j - 1] || graph.instructions[j].scope {
                break;
            }
            if let Some(y) = &candidates[j]
                && partner[j].is_none()
                && y.a == x.a
                && y.b.len() == x.b.len()
                && y.d.iter().all(|register| !x.d.contains(register))
            {
                partner[k] = Some(j);
                partner[j] = Some(k);
                found.push((k, j));
                break;
            }
            let between = &graph.instructions[j];
            if between.flow != Flow::Next
                || scan::names(between.operands).any(|name| x.d.contains(&name))
                || loads.contains(&between.item)
            {
                break;
            }
        }
    }
    if found.is_empty() {
        return Pairs::default();
    }
    // The registers of the pairs that stand (S), until every writer and every reader is shown to
    // keep their state; a pair that one does not is dropped, and S is looked at again.
    let whole = whole_places(&graph);
    let destinations = |&(k, j): &(usize, usize)| {
        let (x, y) = (candidates[k].as_ref(), candidates[j].as_ref());
        x.into_iter().chain(y).flat_map(|c| c.d.iter().copied())
    };
    let mut standing = vec![true; found.len()];
    let (registers, paired) = loop {
        let mut registers: HashSet<&str> = HashSet::new();
        let mut paired: HashSet<usize> = HashSet::new();
        for (pair, ends) in found.iter().enumerate() {
            if standing[pair] {
                registers.extend(destinations(ends));
                paired.extend([ends.0, ends.1]);
            }
        }
        let mut bad: HashSet<&str> = registers.iter().copied().filter(|name| !clean(name)).collect();
        for (k, instruction) in graph.instructions.iter().enumerate() {
            if paired.contains(&k) || copy_into(instruction, &registers, uniform).is_some() {
                continue;
            }
            // Written by something else.
            bad.extend(scan::names(first_operand(instruction.operands)).filter(|name| registers.contains(name)));
            // Read where the 32 lanes may not be together, or by an instruction after which they
            // may not all be at the next one.
            if !whole[k] || instruction.flow != Flow::Next {
                bad.extend(read(instruction, &registers));
            }
        }
        let mut dropped = false;
        for (pair, ends) in found.iter().enumerate() {
            if standing[pair] && destinations(ends).any(|name| bad.contains(name)) {
                standing[pair] = false;
                dropped = true;
            }
        }
        if !dropped {
            break (registers, paired);
        }
    };
    if registers.is_empty() {
        return Pairs::default();
    }
    // Which registers of S may be read after each instruction (liveness, on every way).
    let mut numbered: Vec<&str> = registers.iter().copied().collect();
    numbered.sort_unstable();
    let number: HashMap<&str, usize> = numbered.iter().enumerate().map(|(at, name)| (*name, at)).collect();
    let words = numbered.len().div_ceil(64);
    let set = |names: &mut dyn Iterator<Item = &str>| {
        let mut bits = vec![0u64; words];
        for name in names {
            if let Some(&at) = number.get(name) {
                bits[at / 64] |= 1 << (at % 64);
            }
        }
        bits
    };
    let mut uses: Vec<Vec<u64>> = Vec::with_capacity(n);
    let mut defines: Vec<Vec<u64>> = Vec::with_capacity(n);
    for (k, instruction) in graph.instructions.iter().enumerate() {
        if paired.contains(&k) {
            let own = || candidates[k].iter().flat_map(|c| c.d.iter().copied());
            uses.push(set(&mut own()));
            defines.push(set(&mut own()));
        } else if let Some((destination, source)) = copy_into(instruction, &registers, uniform) {
            uses.push(set(&mut source.into_iter()));
            defines.push(set(&mut std::iter::once(destination)));
        } else {
            uses.push(set(&mut read(instruction, &registers).into_iter()));
            defines.push(vec![0; words]);
        }
    }
    let mut live_in: Vec<Vec<u64>> = vec![vec![0; words]; n + 1];
    let mut changed = true;
    while changed {
        changed = false;
        for k in (0..n).rev() {
            let mut live: Vec<u64> = vec![0; words];
            for &s in &graph.successors[k] {
                for (word, bits) in live.iter_mut().zip(&live_in[s]) {
                    *word |= bits;
                }
            }
            for ((word, used), defined) in live.iter_mut().zip(&uses[k]).zip(&defines[k]) {
                *word = (*word & !defined) | used;
            }
            if live != live_in[k] {
                live_in[k] = live;
                changed = true;
            }
        }
    }
    let mut pairs = Pairs::default();
    for (k, instruction) in graph.instructions.iter().enumerate() {
        if paired.contains(&k) || copy_into(instruction, &registers, uniform).is_some() {
            continue;
        }
        let reads = read(instruction, &registers);
        if reads.is_empty() {
            continue;
        }
        // (A reader is an instruction after which control goes to the next one: see above.)
        let later = &live_in[k + 1];
        let again: Vec<&str> = reads
            .iter()
            .copied()
            .filter(|name| later[number[name] / 64] & (1 << (number[name] % 64)) != 0)
            .collect();
        pairs.before.insert(instruction.item, reads);
        if !again.is_empty() {
            pairs.after.insert(instruction.item, again);
        }
    }
    for (pair, &(k, j)) in found.iter().enumerate() {
        let (Some(x), Some(y)) = (&candidates[k], &candidates[j]) else {
            continue;
        };
        if !standing[pair] {
            continue;
        }
        for (f, a) in x.a.chunks(2).enumerate() {
            let row = (a[0], a[1]);
            if !pairs.rows.contains(&row) {
                pairs.rows.push(row);
            }
            let col = (x.b[f], y.b[f]);
            if !pairs.cols.contains(&col) {
                pairs.cols.push(col);
            }
        }
        pairs.first.insert(graph.instructions[k].item);
        pairs.second.insert(
            graph.instructions[j].item,
            Pair {
                x: x.d.clone(),
                y: y.d.clone(),
                a: x.a.clone(),
                bx: x.b.clone(),
                by: y.b.clone(),
            },
        );
    }
    pairs
}
