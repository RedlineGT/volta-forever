# For AI coding agents

This file is for an agent that was asked either to make a CUDA 13 program run on a Tesla V100
with volta-forever (VF), or to change VF itself. Read it before anything else in the repository.

VF is a young project. Expect that the program in front of you may not work at the first try,
and that the right fix is sometimes in VF. Both are normal outcomes; a wrong result that nobody
noticed is not. The rule that governs every decision here: **refuse rather than guess**. VF
refuses a kernel it cannot run right, and so should you refuse to report "it works" for
something you did not compare with a reference.

## 1. Making a program run

VF places its `libcuda.so.1` in front of NVIDIA's driver. It retargets PTX, translates supported
Turing (`sm_75`) and selected Ampere (`sm_80`) machine code to Volta (`sm_70`), and emulates
some later instructions. It routes libraries such as cuBLAS to their CUDA 12.9 builds. The user
starts a program with the prefix `vf`. [README.md](README.md) has the overview,
[INSTALL.md](INSTALL.md) every option and environment variable, [LIMITS.md](LIMITS.md) what
is known not to work, and `docs/guides/` one worked example for each inference engine.

Steps, in this order:

1. **Check the machine.** A Tesla V100, the NVIDIA R580 driver, Linux x86-64 with glibc 2.34 or
   later. Install with the command of the README, then run `vf doctor` and read every line.
2. **Read the guides.** `docs/guides/` has one worked example for each engine that runs today
   (vLLM, SGLang, llama.cpp-based engines, ZML and others), from a clean Ubuntu to a measured
   result. If your program is there, follow its guide as written before improvising: the
   settings were measured. If it is not, read the guide of the nearest program anyway (same
   framework, same way of building or loading kernels): it shows which settings exist, in which
   order to try them, and what a working run looks like.
3. **Run it plainly first:** `vf PROGRAM ARGS`, or `vf venv DIR` for a Python environment that
   takes ordinary CUDA 13 wheels. Do not set `VF_*` variables you have no reason for.
4. **Read standard error.** Every line that starts with `[volta-forever]` is information meant
   for you:
   - `kernel NAME cannot run on this GPU and is not given to the program: REASON` is a refusal.
     The program gets error 218 (`CUDA_ERROR_INVALID_PTX`) and a NULL handle when it looks the
     kernel up. The reason names what is missing (an instruction with no Volta sequence,
     machine code that is not translated, registers).
   - `emulated in NAME: ...` is logged when VF gives the program a handle to a kernel with
     emulated instructions. Launch it and check its output to verify execution. Measure it
     before drawing conclusions about speed.
   - a line about what a library `is told` is the compute capability that caller sees.
   `VF_LOG_LEVEL=2` prints which images were adapted and what each caller is told.
5. **Libraries.** `vf fetch --all` downloads the CUDA 12.9 libraries VF forwards to and includes
   the CUDA 12 build of NCCL. For NCCL collectives in PyTorch or SGLang, `vf fetch --only libnccl`
   fetches just that library. Direct CUDA peer copies work without NCCL.
6. **Capability.** On a V100, CUDA 13 callers normally see 7.5. Callers with native Volta code
   can keep 7.0. Record which API and which caller reported the value. With `VF_CAPABILITY=8.0`,
   NVML can report 8.0 while PyTorch's CUDA runtime still reports 7.5 because it uses translated
   `sm_75` kernels. Adding `VF_CAPABILITY_ALL=1` lifts that cap for the CUDA runtime and the code
   that uses it: PyTorch's attention then fails (its kernels for 8.0 are stubs in that code, and
   are refused). Libraries that ask for themselves without that runtime (cuDNN, cuFFT, cuSOLVER)
   are still told what their code allows. Use
   the override only in a
   configuration verified by the engine guide and compare its outputs. Triton takes PyTorch's
   value: told 7.5, its `tl.dot` uses no tensor cores (right, but many times slower), and only
   `VF_CAPABILITY_ALL=1` changes that in a program with PyTorch. [Capability rules](LIMITS.md#the-capability-told-with-vf_capability)
7. **Prefer PTX.** If you build the program or its extensions yourself, build them with PTX
   retained. Use `TORCH_CUDA_ARCH_LIST="7.5+PTX"` for a PyTorch extension or add
   `-gencode arch=compute_75,code=compute_75` to the `nvcc` command. VF can then retarget the
   embedded PTX for the driver to compile on Volta.
8. **Verify.** "It printed something" is not a result. Compare with a reference: the same
   program on the CPU, a native CUDA 12 build, or a newer GPU. For a language model, compare
   greedy tokens. Report to the user exactly what you compared and what you did not.
   For engine throughput, follow the [35B benchmark protocol](docs/guides/benchmarks/README.md).
   Compare the same checkpoint and quantization on the same GPU with speculative decoding off.
   Report median prefill and decode rates separately. Reserve global `ptxas -O0` and forced
   eager mode for diagnosis. Benchmark the verified configuration with the engine's fast paths.
   To tell whether a difference comes from an opt-in of the layer, run again with `VF_SAFE=1`
   (exactly `1`): the switches whose results are not bit-identical to the default's are then not
   used, nor the pairs of products of translated machine code (whose last bits depend on a
   kernel's registers), and the layer says what it turned off. It does not change the capability
   told.
   [Safe mode](INSTALL.md#environment-variables)

When it does not work, say which of these it is: a setting (fix it and tell the user), a
documented limit (quote LIMITS.md), or something VF lacks (section 2). Do not paper over a
refusal by patching the program to skip the kernel unless the user asks for that.

## 2. Changing VF

Read [CONTRIBUTING.md](CONTRIBUTING.md) whole: it is short and every rule in it was paid for.

**Fix the general problem, not your program.** A change to VF must make a class of programs
work: an instruction lowered for every kernel that uses it, a driver entry point that behaves
as NVIDIA's does for every caller, a kind of image that is read right wherever it comes from.
Select kernel translations from the instructions and their semantics. Keep engine, library and
kernel names out of those decisions. Avoid hand-written replacements for one engine's kernel
and VF settings specific to one engine. The launcher uses library names
(SONAMEs) to route compatible implementations, as it does for `libnccl.so.2`.
Show the underlying problem with a small test that runs independently of the engine. Put
engine-specific settings and commands in its guide under `docs/guides/`.

The rules agents break most often:

- **A test must be able to fail.** Every fix comes with a test that fails before it and passes
  after, and the commit says it was seen to fail.
- **Two translators, one result.** Keep `crates/vf-sass` and the Python reference in agreement
  (`reference/translate.py`, with `reference/ampere.py` for `sm_80`). Run `tests/sass/compare.sh`.
- **Independent validation covers `sm_75` machine code.** Add a rule in `reference/validate.py`
  and a negative control in `reference/controls.py` for a new `sm_75` rewrite. For `sm_80`,
  check Python/Rust agreement, run the unit tests and verify the affected kernels on a V100.
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
