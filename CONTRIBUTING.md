# Contributing

## Build and test

```sh
scripts/build.sh                 # dist/lib, dist/bin
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

`tests/README.md` says what the other suites need. Without a GPU, the one to run after any
change to the machine-code translator, and the one after a change to how the proxy takes and
launches kernels:

```sh
cargo build --release -p vf-cli
scripts/fetch-all.sh denvdis     # the instruction descriptions that reference/ reads
tests/sass/compare.sh            # the Rust translator against the Python reference
tests/proxy/fit_api.sh           # the second translation at launch, by every way to it
tests/proxy/lifecycle.sh         # what the proxy keeps of modules and contexts, fork, the cache
dash tests/launcher/launcher.sh  # the launcher (and under bash --posix)
```

`tests/README.md`, "What the CI runs", lists what every push runs; the host tests of
`tests/proxy/` that it does not run are listed there too.

## What a change must keep

**Refuse rather than guess.** A kernel that the layer cannot run right is refused when the
program asks for it, with a reason that names what is missing. A translation or an emulation
that may be wrong is not offered.

**Two translators, one result.** `reference/translate.py` (with `reference/ampere.py` for
sm_80 code) is the reference of the machine-code translator and `crates/vf-sass` its
implementation in the layer: for every input they give the same bytes, the same map, the same
`MAP.lives` and the same refusals (`tests/sass/compare.sh` on real code, `tests/sass/fuzz.py`
on damaged code). A change to what is translated is a change to both.

**The validator is another program.** `reference/validate.py` checks a translation without
sharing the translator's decisions. A new kind of rewriting needs its rule there and, in
`reference/controls.py`, a translation that is wrong on purpose and that the rule finds: a
check that was never seen to fail has not been shown to check (`docs/validation.md`).
`tests/sass/controls.sh` runs every control, each in a process of its own, and says which
change nothing: a control must be seen to change the translation and to be objected to for its
own reason. The translation of sm_80 code has no rule there yet: it is not independently
validated (`docs/validation.md`, section 8), and a change to it rests on `compare.sh`, unit
tests and the GPU cases.

**A test must be able to fail.** A new mechanism comes with a test that fails when the mechanism
is turned off, and the commit says that it was seen to. A test that computes on constants, or
that compares a result with itself, is not a test.

**The instruction tables are generated.** `crates/vf-tables` is written by
`reference/generate.py` (then `cargo fmt -p vf-tables`) from the descriptions that
`scripts/fetch-all.sh denvdis` fetches, and is never edited by hand. It is under the LGPL 2.1
(`crates/vf-tables/README.md`); everything else is under the Apache License 2.0, and
contributions are taken under the licence of the files they change.

**An emulation states its arithmetic.** What is exact, what is rounded and how, what becomes of
NaN, infinities and overflow, and against what it was compared (a host model in
`tests/kernels/tcrun.c`, a native build, results from the hardware that has the instruction):
`docs/emulations.md` is kept true.

**A figure says what was measured.** On what, against what, how many runs, in one session or
not, and when. Two figures that are compared come from one session on one GPU.

**Nothing of NVIDIA's in the repository.** No library, header, documentation or tool. The test
kernels are this project's sources and what NVIDIA's compilers made of them; the forwarding
libraries are generated on the user's machine from the toolkits that the user fetched. One piece of
compiler output goes into the release archive: `crates/vf-fwdgen/bf16tc/kernels.fatbin`, the sm_70
machine code of the bf16 cuBLAS runtime, made by `scripts/build-bf16-kernels.sh` from
`crates/vf-fwdgen/bf16tc/kernels.cu`. That source calls no routine of NVIDIA's libraries or headers
(only the one-instruction intrinsics of the language), writes its own conversions, and rounds only
through explicitly rounded operations; the script refuses PTX in the result and external functions
in the intermediate PTX, and `--check` rebuilds it to the same bytes. Change the source, rebuild,
and commit the source, `kernels.fatbin` and `kernels.manifest` together.

## Reporting a wrong result

A program that computes something else through the layer than on the CPU or on a newer GPU is
the report that matters most. Useful with it: the program and its version, the driver, the
`VF_*` variables that were set (`VF_CAPABILITY` above all), what `VF_LOG_LEVEL=2` prints
(which images were adapted or translated, which kernels have emulated instructions, what each
caller is told), and, for machine code, the directory that `VF_TRANSLATE_DUMP=DIR` fills with
every unit of machine code given to the translator: `reference/validate.py` can then be run on
its sm_75 units.
