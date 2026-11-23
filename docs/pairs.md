# Pairs of products on whole rows: the design

How the machine-code translator computes two of Turing's tensor-core products that share their A
with two of Volta's `m8n8k4` on whole rows of shared memory, where it would otherwise compute
each product by itself with its operands gathered from other lanes (`docs/translator.md`,
"Tensor-core products in machine code"). The arrangement is called Q below. In a loop written
by hand it runs at 1.2 to 1.4 times the time of a loop written for Volta, where products
translated one by one take 5.3 to 7.6 times (`docs/performance.md`, "Where the time goes"); its
results are bit for bit those of the one-by-one translation on exact inputs, and no further
from an RTX 5090's on random ones (section 9).

This is the specification that three programs follow: `reference/pairing.py` and
`crates/vf-sass/src/pairing.rs` make the plan of a function (section 11), the two translators
write the sequences, and `reference/qcheck.py` checks a plan against the original function
without sharing the planner's decisions. The words "must" below are what a translation has to
satisfy and what the checker verifies on its own. Where the translator has a choice, the text
says "the translator chooses".

Both translators make pairs by default; `VF_PAIRS=0` turns them off, and every product is then
translated by itself. `VF_PAIRS_SAY=1` makes the Python reference say on standard error where it
gives pairs up and why. Pairs are made for sm_75 code only: the sm_80 forms that the translator
accepts have no tensor-core instruction.

## 1. Terms

- A lane is `L = 16h + 8sr + 4sc + 2k1 + k0` (five bits). Turing's layouts go by `g = L >> 2` and
  `c = L & 3`; Volta's instruction makes one 8x8x4 product for each of the four sets of eight lanes
  `{16h + 4q + k}` with `q = 2sr + sc`, `k = 2k1 + k0`.
- **Aligned instruction**: `HMMA.1688` and `LDSM` of the original (PTX `mma.sync.aligned`,
  `ldmatrix.sync.aligned`): a program may run one only with the 32 lanes of the warp active
  together. The translation of a product by itself already rests on this (its sequences shuffle
  between lanes).
- A matrix load `LDSM.4` reads 32 rows of 16 bytes (8 halves): lane t gives the address of row t
  (`Ra` of lane t, plus the uniform part and the offset of the instruction). "Row t" below.
  - Used as A: rows 0..15 are the A fragment 0 (Turing registers `Rd`, `Rd+1`), rows 16..31 the
    fragment 1 (`Rd+2`, `Rd+3`): `A_a[R][k]` is half k of row `16a + R`.
  - Used as B, plain: B fragment b (Turing register `Rd+b`) is rows `8b..8b+7`: `B_b[k][n]` is
    half k of row `8b + n`. Transposed (`LDSM.T`): `B_b[k][n]` is half n of row `8b + k`.
  - The **pair p** of a B load is the fragments 2p and 2p+1 (registers `Rd+2p`, `Rd+2p+1`).

## 2. The arrangement (checked by `reference/pairs_model.py`, both forms)

For an A fragment a and a pair p, with f32 accumulators updated in place:

| What | A thread holds |
|---|---|
| A operand | the four words of row `16a + kA`, `kA = 8k1 + 4sr + 2h + k0` |
| B operand, plain load | the four words of row `16p + kB`, `kB = 8h + 4sc + 2k1 + k0` |
| B operand, transposed load, half κ of k | the two words at byte `cB = 8sc` of row `16p + kBT + 4κ`, `kBT = 8h + 2k1 + k0` |

Two of Volta's instructions, the first on words 0-1 of A and the first half of B (κ = 0), the
second on words 2-3 and the second half: `HMMA.884.F32.F32` with A by rows, and B by columns for a
plain load, by rows for a transposed one. Of the eight result registers, register i of lane L is
register `i & 3` of the Turing accumulator of the product on B fragment `2p + (i >> 2)`, as lane
`pi(L)` holds it: `pi(L) = 16sr + 8h + 4k0 + 2sc + k1`.

So a Turing accumulator stays in its own four registers and is **permuted between lanes** by one
permutation, the same for every accumulator and both forms of B. Below a register is in state
**Q** when it holds an accumulator register so permuted, **T** when it holds what the program
means (Turing's layout, or any ordinary value), **Z** when every lane holds zero (both at once).
`T[t] = Q[pi^-1(t)]`: one `SHFL.IDX` a register each way, with the index `pi(L)` to go to Q and
`pi^-1(L) = 16*b3 + 8*b4 + 4*b1 + 2*b0 + b2` (bits of L) to come back.

## 3. What is translated this way

**A whole-row load** is an `LDSM.4`, not under a predicate, 16-bit elements, destination `Rd` a
multiple of 4, plain or transposed. It puts the rows of section 2 in registers (section 4). When
what it gives in Turing's layout may be read, on some way, by anything but the products of pairs
below (a way: along the instructions, where a write of the register under no predicate ends it, and
a write under a predicate hides the load from readers under the same predicate until that predicate
is written), the load is marked **also**: the registers `Rd..Rd+3` then get Turing's layout as
before, beside the rows.

**A pair of products** is two `HMMA.1688.F32` i < j of one function, not under a predicate, each
with a destination `D` of four registers (even) and an accumulator `C` it adds to: its destination
(in place), four other registers, or zero (`RZ`); such that:

1. no instruction of (i, j] is entered otherwise than from the one before it, and no instruction
   of [i, j) transfers control (a call included);
2. they have the same A register `Ra`, and their B registers are the two of one pair
   (`{RdB + 2p, RdB + 2p + 1}`);
3. their eight destination registers are distinct, and none is `Ra`, `Ra+1` or a B register of the
   pair; the accumulator each product reads is its own destination, or has no register of either
   destination (the second product does not add to what the first one gives);
4. at i, every definition that reaches (in the original) `Ra` and `Ra+1` is a whole-row load with
   one destination `RdA` (`Ra = RdA + 2a`), all plain, used as A by every pair they reach; and every
   definition that reaches `RdB + 2p` and `RdB + 2p + 1` is a whole-row load with destination `RdB`,
   all plain or all transposed, used as B by every pair they reach; `RdA` is not `RdB`. On no way
   to i is one of these four registers as the function was entered with it. At j, the same holds
   of the three registers the product at j reads (`Ra`, `Ra+1`, its B register): the original
   product at j computes on them, whatever the translation reads. (The B register of the product
   at i may have been written between i and j: PyTorch's kernels reuse it.)
5. no instruction strictly between i and j reads or writes one of the eight destination
   registers, none writes the accumulator that the product at j reads, and none is a matrix load
   with destination `RdA` or `RdB`. (The accumulator the product at i reads may be written after
   it: compilers that rename accumulators load the next operands there.)

The product at i is done as the first of Volta's two instructions (κ = 0) **for both products**:
it reads the accumulators of both (the one of the product at j where that product would read it
later: 5 keeps it the same) and writes both destinations; the product at j is done as the second
(κ = 1), in place on both destinations. Between them the two destinations hold partial sums,
which 5 keeps unseen. x is the product whose B register is the even one, y the other: steps 0 to
3 write `Dx`, `Dx+2`, `Dy`, `Dy+2` (pairs of registers), the first instruction from `Cx`, `Cx+2`,
`Cy`, `Cy+2` (or zero).

In the 249,016 products of the held-out corpus, 6.5 % of those with f32 results do not add in
place (4.3 % to zero, 2.2 % to other registers); a loop that a compiler unrolled has them at
each turn (4 of the 40 products of the test loop: with them left out of pairs and converted at
every turn, it took 0.95 ms where it takes 0.48).

Everything else (other loads, other products, halves accumulators for now) is translated as
before; such a product is, for what follows, an ordinary instruction that reads its operands and
accumulators (and a reader of Turing's layout: the loads that reach it are **also**).

## 4. Registers

For each destination `Rd` of whole-row loads in a function, registers of the translation `E(Rd)`
that nothing but those loads writes. Each row is four words: the first two are the first half of
k, the last two the second half. For the fragment or pair q (0 or 1):

| | first half of k | second half of k |
|---|---|---|
| no load with that destination is **also** | `Rd+2q`, `Rd+2q+1` (the program's own) | `E+2q`, `E+2q+1` |
| one is | `E+2q`, `E+2q+1` | `E+4+2q`, `E+5+2q` |

So with four registers of the translation, a fragment's first half is where the program has that
fragment: the first product of a pair reads the program's own registers, the two of its A and
the two B registers of the pair (condition 4 at i is about those four), and the second product
reads registers of the translation only. With eight, the program's registers are left to
Turing's layout and both products read registers of the translation.

**Sharing.** Destinations whose rows are never wanted at the same time may have the same
registers `E` (PyTorch's attention kernels have eight destinations in two loops, two rows in
flight at a time, and would not fit in 253 registers otherwise): the plan says which (`share`,
section 11). What must then hold, and is all that is asked of a plan: for every product of a pair
and each of its two destinations whose rows it reads in registers of the translation, on every
way to the product the last whole-row load among the destinations that have those registers is a
load of the product's destination.

Kept from the entry of the function, computed once from the lane, in registers nothing else
writes: `kA` and `kA + 16`, `kB` and `kB + 16` (if a plain B is loaded), `kBT` and `cB` (if a
transposed one), `pi(L)` and `pi^-1(L)` (if a conversion uses them). Scratch for a load's
addresses: the scratch registers the matrix sequences already have.

## 5. Sequences

At a whole-row A load (`U`: the uniform part of the address, if any; `off`: the offset):

    s0 = SHFL.IDX(Ra, kA)                 the address register, as the lane with the row of fragment 0 has it
    s1 = SHFL.IDX(Ra, kA + 16)            of fragment 1
    s0 += U, s1 += U                      when there is a uniform part
    first and second half of fragment 0 = 8 bytes at [s0 + off] and at [s0 + off + 8]
    first and second half of fragment 1 = 8 bytes at [s1 + off] and at [s1 + off + 8]

followed, when the load is **also**, by the sequence that stands for a load by itself (after: that
one writes the program's registers, and the address register may be one of them). A plain B
load: the same with `kB`. A transposed one: four rows (the lanes `kBT + 4κ + 16p`, computed into
scratch registers first), each address plus `cB` (plus U: one sum, made before the shuffles), 8
bytes each into the half κ of pair p. The 8 bytes are one 64-bit load when rows are taken to be
aligned on 16 bytes (the policy the program-text lowering has by default: `aligned_rows`; a wide
load at an address that is no multiple of its size faults, measured), two 32-bit loads otherwise.

The shuffles come first and on the address register as it is, what is added after them, and
`kA + 16` is kept and not computed at each load: with this and the scoreboards of section 8,
PyTorch's attention kernel went from 2.995 ms to 2.85.

At a product of a pair, half κ: four steps

    HMMA.884.F32.F32.STEPn  D(n), A half κ (.ROW), B half κ (.COL, or .ROW for a transposed B), D(n)

A conversion of register r: `SHFL.IDX r, r, K` with K the kept `pi(L)` (to Q) or `pi^-1(L)` (to T),
under a guard when it is put before a conditional branch for one of its two ways (section 7).

## 6. States

Each register of the original has, at each place, a state among Z, T, Q (section 2), by this:

- at the entry of the function every register is T;
- a product of a pair **requires** Q or Z of the accumulator registers it reads in the translation
  (the first product of a pair: those both products add to; the second: the eight destination
  registers), and leaves the eight destination registers Q;
- any other instruction that reads a register, or writes it under a predicate, requires it T or Z,
  and leaves what it writes T (Z when it is an instruction, under no predicate, that writes zero);
- a call, a return, and a branch whose targets are not known require every register T or Z;
- where two ways meet with a register Q on one and T on the other, the register's state is not
  known, and nothing may read it before it is written whole.

A conversion changes the state of its register: to Q requires T or Z, to T requires Q or Z.

## 7. Where a conversion may be

A conversion shuffles between lanes: it must be at a place that is **whole**: where, whenever a
lane runs it, the 32 lanes run it together. Sufficient, and all that is used:

- **(lead)** a place from which straight-line code reaches an aligned instruction: no conditional
  branch, no instruction that parks or removes lanes (`BSYNC`, a guarded `EXIT`/`RET`/`KILL`/`BREAK`),
  no call on the way; unconditional branches are followed (`BRA` and `BRA.U` under no predicate:
  cuBLAS jumps over the other arm of a choice with `BRA.U`; `BRA.DIV` and `BRA.CONV` are taken
  or not by what the lanes are doing). The lanes that are there all reach the aligned
  instruction together, which needs 32.
- **(after)** what follows an aligned instruction in straight-line code, up to and including the
  next branch.
- **(uniform)** a conditional branch at a whole place, one of whose two ways leads (as in
  **lead**) to an aligned instruction, is taken by all lanes or by none: both ways start whole.
- a place that is entered several ways is whole when every way into it is (or by **lead**).

Conversions are put: before an instruction, at a whole place, for what that instruction requires;
after an instruction that falls into a place entered other ways too; before an unconditional
branch; for the way taken of a uniform conditional branch, before it under its condition; for the
other way, after it. The translator chooses among these (it converts to T as soon as no way from
there uses the register as an accumulator of a pair before something else reads it, and to Q on
the ways into a loop of products); the validator only checks that every requirement of section 6
holds on every way, and that every conversion is at a whole place.

When a requirement cannot be met by conversions at whole places, the products concerned are not
translated as pairs (section 3 no longer holds for them) and everything is computed again.

## 8. Schedule

Measured on the hand-written loop with its scheduling fields changed: every
product waiting for one scoreboard costs nothing; eight cycles after every last step cost 22 % at
low occupancy; one scoreboard for all loads 23 %; the shuffles of addresses on one scoreboard 15 %;
all together 38 % at one block a multiprocessor and 2 % at eight.

- Steps two cycles apart; the last step of each product of a pair names the sequence scoreboard,
  and the first step of each waits for it.
- A load of a row names, for what it loads, the scoreboard the original load named: the
  program's readers wait for it as they did. When the original named none, the sequence
  scoreboard: the products of pairs, the only readers of rows, all wait for that one.
- The shuffles of addresses name another scoreboard (4; the sequence scoreboard, 5, when the
  original load named 4 for its result), the first instruction that reads a shuffled address
  waits for it, and every load says on it when it has read its address (a scratch register, which
  the next sequence writes): what follows the last load waits for that. So loads do not wait for
  the products before them, nor what follows a load for them.
- The last instruction of a sequence stalls what the original instruction's stall leaves after
  the sequence's own instructions (after a product's four steps: the original's less six, two at
  least).
- Before the first of a run of conversions: an instruction that does nothing, waits for every
  scoreboard and stalls fifteen cycles, the most an instruction can. A conversion reads its
  register where no instruction of the program did, so no wait of the program is known to cover
  it: an accumulator that a load of the program is still filling is waited for by its reader,
  further on (accumulators taken from memory and converted on the way into a loop came out
  wrong on a V100 without this); a product's result is not readable by anything but a product
  for eight cycles, wait or no wait (`docs/translator.md`, "Timing rules"). The
  conversions name the sequence scoreboard and what follows waits for it.
- What the program's instruction waited for, the first instruction of the sequence that stands
  for it waits for, and only the first: on every instruction of a load it would make each load
  of a row wait for the one before.

## 9. Numbers

Each result is `(C + sum over k in 0..3) + sum over k in 4..7`, each sum by Volta's instruction;
the translation of a product by itself adds the even k then the odd; Turing's own instruction is
neither. On 327,680 random results compared with an RTX 5090's: a product by itself 47.6 %
identical and 41.2 % one unit of the last place away; Q 47.2 % and 41.7 %; the same tails.

## 10. What rests on what

1. The original runs aligned instructions with whole warps (PTX's rule for `.aligned`).
2. The reference layouts of `tests/kernels/tcrun.c`, each checked on a V100.
3. Rows of a matrix load aligned on 16 bytes when 128-bit loads are used (PTX's rule for
   `ldmatrix`; `VF_ALIGNED_ROWS=0` turns the assumption off).
4. The function is entered at its start only, as for every function with matrix instructions.
5. Lanes that part at a conditional branch run together again only after a `BSYNC`: nothing else
   promises it, so a compiler that must have 32 lanes at an aligned instruction (1) cannot have a
   branch that parts lanes with no `BSYNC` between it and that instruction on one of its ways.
   This is what **uniform** (section 7) rests on.

## 11. The plan as data

What the translator decides for a function, and what an independent checker is given with the
original function (indices are instructions of the original, registers are numbers):

    plan = {
        'loads': {i: {'role': 'A' | 'B' | 'BT', 'rd': Rd,           # whole-row loads (BT: a transposed B)
                      'also': bool}},                                # Turing's layout made as well (section 3)
        'sites': {i: {'half': 0 | 1, 'partner': j,                   # products of pairs
                      'x': Dx, 'y': Dy,                              # the destinations of the pair
                      'cx': Cx, 'cy': Cy,                            # what each adds to: registers, or None for zero
                      'ra': RdA, 'a': 0 | 1, 'rb': RdB, 'p': 0 | 1}},
        'conversions': [(i, 'before' | 'after', register, 'Q' | 'T', None | (predicate, negated))],
        'share': {Rd: Rd'},                                          # the destination whose registers of rows Rd has (section 4)
    }

`before`: among what is put before instruction i (a guard, when given, is the condition under
which the conversion runs: the way taken of the conditional branch i, or the other with `negated`
turned). `after`: after instruction i, run only when i falls through. Everything that is not in
`loads` or `sites` is translated as before.

## 12. Companions: a load's addresses without their shuffles (an experiment, in the reference only)

`VF_COMPANIONS=1`, read by the Python reference alone (`pairing.companions`, and
`whole_rows`, `accompanied`, `again` in `reference/translate.py`); off by default, and with it
off the reference writes what the Rust translator writes. **The validator, the checker of plans
and the Rust translator know nothing of it: not to be turned on outside experiments.**

Measured: what a warp of the translation waits for, where the same loop written by
hand does not, is the shuffles of each load's address register: 19 % of the loop kernel's time at
one block a multiprocessor, 13 % of PyTorch's attention kernel.

**What it rests on.** A shuffle commutes with every operation done lane by lane: if
`Ra = f(X, constants)` in every lane, then `Ra` as lane K has it is `f(X as lane K has it,
constants)`.

**Terms.** A **root** is a register X from which loads take addresses. A **companion** of X for
a row of a role (A: 2 rows, B: 2, a transposed B: 4) is a register of the translation that holds X
as the lane of that row has it (`SHFL.IDX companion, X, kept lane`). It **stands** from the
shuffle that makes it until X is written by anything (a guarded write, a call that may write it).

**A load through a root**, two forms:

- *direct*: its address register is the root; each row's address is the companion (plus the
  uniform part, plus the byte of a transposed B: additions at the load, as before).
- *by an instruction done again*: its address register `Ra` has one definition that reaches the
  load, an instruction under no predicate that reads one register X (not `Ra`) and constants,
  writes `Ra` and nothing else, and is plain arithmetic (`LOP3`, `IMAD`, `LEA`, `IADD3`, `SHF`,
  `MOV`); and no way from that definition to the load, short of passing the definition again, has
  a write of X. The load does that instruction on each companion of X, into the row's scratch
  register.

**Which loads.** Only where shuffles leave a loop: the load is on a cycle that has no write of
the root. Roots are taken greedily, the one that serves the most loads first; a load that comes
round only through a place where its root's companions are made keeps its shuffles.

**Where companions are made.** Followed from the entry of the function and from every write of
the root, along every way: the first place that is whole (section 7) and from which a load
through the root is reached with no write of it gets the shuffles, before its instruction (after
the instruction before it, when that one alone brings the root without companions, is a whole
place, and other ways come in). Past such a place the companions stand, through places that are
not whole too. The sequence: an instruction that does nothing, waits for every scoreboard and
stalls fifteen cycles (the root is read where no instruction of the program read it), the sums of
lanes for a transposed B, the shuffles; what follows waits for them. A load therefore waits for
nothing: its companions arrived long before.

**Registers**: 2 or 4 for each root and role, after everything else; when the function has not
that many left, companions are dropped first (then pairs, as before).

**Measured on a V100** (the reference's translation given to a build of the proxy made for the
experiment; results the same to the last value in every case; 2026-11-18):

| | pairs | pairs and companions | by hand / native |
|---|---:|---:|---:|
| loop kernel, 2000 turns, 80 blocks | 0.481 ms | **0.388** | 0.335 |
| 160 blocks | 0.630 | **0.563** | 0.493 |
| 640 blocks | 2.05 | **1.93** | 2.07 |
| PyTorch `attention.f16` | 2.79, 2.80 | **2.68, 2.67** | 1.69 |

The loop kernel's results are those of its references for 1 to 100 turns; `k_pairs`: 34 kernels,
0 wrong; attention forward and backward: the counts of identical values of the run without
companions.

**What the attention kernel shows** (`fmha_cutlassF_f16_aligned_64x64_rf`, 20 loads): 12 go
through companions, of `R0` (written once, 138 instructions before the first whole place on the
way, which is where its four companions are made: once a tile) and of `R2` (the four transposed
loads of the second loop). The other 8 are the second loop's loads whose address register is made
in the loop from two registers the loop writes (`IMAD R, R20, 0x10, R3`): not handled. The gain is
4 % and not the 13 % of the samples because:

- 6 of the 12 do an instruction again (`R0 ^ constant`) and then add the uniform part: two
  instructions of arithmetic a row where there were a shuffle and an addition, and the scheduler
  keeps 4 cycles from the first to the second and 8 from the second to the load (the distances
  native code keeps from arithmetic to arithmetic and to a memory instruction).
- The 8 cycles from the addition of the uniform part to the load are paid by every load, with or
  without companions.
- The kernel has 15 registers left of 253 with pairs: 8 go to these companions.

**Not built, in the order they would pay:**

1. The root that is the address register itself, when its definition is outside the load's
   loop, before the instruction done again (`R2` of the first loop is made once before it).
2. The uniform part (and the byte of a transposed B) added once when the companions are made,
   where every load of a root and role has the same: such a load is then its loads alone.
3. An addition that a memory instruction waits 8 cycles for, issued earlier in straight-line code
   where nothing writes what it reads: every memory instruction with a uniform part of address or
   a stride has such a wait (12 % of this kernel's samples are those two labels), with or without
   tensor cores. A change of the scheduler, to be designed with the validator.
4. Roots written in the loop by arithmetic on registers that are the same in every lane (the
   loop's counters): needs knowing which registers are.
5. The validator, the checker of plans (which must be told the roots, the companions' registers,
   where they are made and which loads go through them, and check that companions stand at every
   such load on every way), the controls, the Rust translator.
