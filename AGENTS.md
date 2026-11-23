# For AI coding agents

This file is for an agent that was asked either to make a CUDA 13 program run on a Tesla V100
with volta-forever (VF), or to change VF itself. Read it before anything else in the repository.

VF is a young project. Expect that the program in front of you may not work at the first try,
and that the right fix is sometimes in VF. Both are normal outcomes; a wrong result that nobody
noticed is not. The rule that governs every decision here: **refuse rather than guess**. VF
refuses a kernel it cannot run right, and so should you refuse to report "it works" for
something you did not compare with a reference.

## 1. Making a program run

What VF is, in one paragraph: a `libcuda.so.1` that sits in front of NVIDIA's driver, retargets
PTX and translates Turing (`sm_75`) machine code to Volta (`sm_70`), emulates some later
instructions, and sends libraries such as cuBLAS to their CUDA 12.9 builds. The user starts a
program with the prefix `vf`. [README.md](README.md) has the overview,
[INSTALL.md](INSTALL.md) every option and environment variable, [LIMITS.md](LIMITS.md) what
is known not to work, and `docs/guides/` one worked example for each inference engine.

Steps, in this order:

1. **Check the machine.** A Tesla V100, the NVIDIA R580 driver, Linux x86-64 with glibc 2.34 or
   later. Install with the command of the README, then run `vf doctor` and read every line.
2. **Look for a guide.** If `docs/guides/` has the program, follow it as written before
   improvising: the settings there were measured.
3. **Run it plainly first:** `vf PROGRAM ARGS`, or `vf venv DIR` for a Python environment that
   takes ordinary CUDA 13 wheels. Do not set `VF_*` variables you have no reason for.
4. **Read standard error.** Every line that starts with `[volta-forever]` is information meant
   for you:
   - `kernel NAME cannot run on this GPU and is not given to the program: REASON` is a refusal.
     The program gets error 218 (`CUDA_ERROR_INVALID_PTX`) and a NULL handle when it looks the
     kernel up. The reason names what is missing (an instruction with no Volta sequence,
     machine code that is not translated, registers).
   - `emulated in NAME: ...` means the kernel runs, slower than on the GPU it was built for.
   - a line about what a library `is told` is the compute capability that caller sees.
   `VF_LOG_LEVEL=2` prints which images were adapted and what each caller is told.
5. **Libraries.** `vf fetch --all` downloads the CUDA 12.9 libraries that VF forwards to.
   Anything that uses several GPUs through PyTorch needs `vf fetch --only libnccl`.
6. **Capability.** By default a CUDA 13 program is told 7.5. Some programs want 8.0 for their
   Ampere code paths: `VF_CAPABILITY=8.0` tells that to callers whose kernels VF can give for
   it, and keeps telling 7.5 to the others (PyTorch, cuDNN), saying so.
   `VF_CAPABILITY_ALL=1` tells everyone, and is dangerous: a library told 8.0 while its code
   comes from `sm_75` machine code can return wrong results with no error (PyTorch's own
   attention, cuDNN convolutions). Use it only as a guide does, and compare outputs. Read
   "The capability told with `VF_CAPABILITY`" in LIMITS.md first.
7. **Prefer PTX.** If you build the program or its extensions yourself, build them with PTX
   kept (for instance `TORCH_CUDA_ARCH_LIST="7.5+PTX"`, or nvcc's `-gencode arch=compute_75`):
   the driver then compiles real Volta code, which avoids the limits of machine-code
   translation.
8. **Verify.** "It printed something" is not a result. Compare with a reference: the same
   program on the CPU, a native CUDA 12 build, or a newer GPU. For a language model, compare
   greedy tokens. Report to the user exactly what you compared and what you did not.

When it does not work, say which of these it is: a setting (fix it and tell the user), a
documented limit (quote LIMITS.md), or something VF lacks (section 2). Do not paper over a
refusal by patching the program to skip the kernel unless the user asks for that.

## 2. Changing VF

Read [CONTRIBUTING.md](CONTRIBUTING.md) whole: it is short and every rule in it was paid for.
The ones agents break most often:

- **A test must be able to fail.** Every fix comes with a test that fails before it and passes
  after, and the commit says it was seen to fail.
- **Two translators, one result.** `reference/translate.py` and `crates/vf-sass` give the same
  bytes; a change to one is a change to both (`tests/sass/compare.sh`).
- **The validator is another program.** A new kind of rewriting needs its rule in
  `reference/validate.py` and a control in `reference/controls.py`.
- **An emulation states its arithmetic** in `docs/emulations.md`, and its expected values come
  from the hardware that has the instruction, or from the PTX manual with that said.
- **Nothing of NVIDIA's in the repository.**

Build and check, without a GPU:

```sh
scripts/build.sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

[tests/README.md](tests/README.md) lists the host suites (run the ones that touch what you
changed) and the GPU suites. If you have a V100, `tests/gpu/run.sh` is the minimum after a
change to the proxy or to a lowering; say in the pull request what you ran and on what, and
say plainly what you could not run. [docs/architecture.md](docs/architecture.md) is the map of
the code.

Where things usually are:

| Symptom | Look in |
|---|---|
| a PTX instruction has "no Volta sequence" | `crates/vf-ptx/src/lower/`, `docs/emulations.md` |
| machine code "is not translated" | `crates/vf-sass`, `reference/`, `docs/translator.md` |
| a library should go to its CUDA 12 build | `scripts/vf`, the forwarding libraries, `INSTALL.md` |
| a caller is told the wrong capability | `crates/vf-adapt`, `crates/vf-proxy/src/callers.rs` |
| a driver entry point behaves unlike NVIDIA's | `crates/vf-proxy/src/hooks.rs` |

## 3. Sending the work back

- One pull request per problem, against `main` of `nbald/volta-forever`, with: the program and
  its version, the driver, the `VF_*` variables set, the `[volta-forever]` lines that showed
  the problem, the test you added, and what you ran.
- Ask the user before you open a pull request or an issue in their name, and before you
  include anything from their machine (paths, model names, logs).
- If you cannot fix it, open an issue with the same information and the smallest input that
  shows it (`VF_TRANSLATE_DUMP=DIR` saves the machine code given to the translator).
- Never weaken a refusal to make a program start. A refusal that is wrong is fixed by making
  the kernel run right and proving it.
