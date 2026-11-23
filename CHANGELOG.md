# Changes

## Unreleased

### Fixed

- With `VF_CAPABILITY` at 8.0 or later, a program whose kernels VF can only give from their sm_75 machine code (PyTorch, cuDNN, cuFFT, cuSOLVER; through the CUDA runtime, the libraries that use it) is still told 7.5, with a line that says so: told 8.0 it computed wrong results with no error, and from 9.0 faulted the GPU. Programs whose code is PTX are told what was asked, as before. cuDNN's graph library is told what the code of its engines allows.
- The capability told to a main program no longer depends on how it was started (by name through `PATH`, after a change of directory): the program is read through `/proc/self/exe`.
- In a machine with a V100 and a later card, modules loaded into a context on the later card are given to the driver as the program gave them.
- A refused kernel lookup (`cuModuleGetFunction`, `cuLibraryGetKernel`) leaves NULL in the handle, as NVIDIA's driver does for a name it does not have; the error is still 218.
- Kernels that cannot run on the GPU, in the inputs of the driver's linker (`cuLinkAddData`), are refused at their lookup in the image `cuLinkComplete` makes, and in copies of it, with the reason; its kernels with emulated instructions are announced. They used to reach the program as code that traps.
- A process made by `fork()` that ends with `exit()` no longer hangs in the layer's reports at exit; these reports are printed only by the process that loaded the layer.
- When a context is destroyed (`cuCtxDestroy`, the last `cuDevicePrimaryCtxRelease`, `cuDevicePrimaryCtxReset`, as on `cudaDeviceReset`), the layer drops what it kept of the modules of that context (adapted images, kernels, originals kept for a second translation, the second translations) instead of keeping them for the life of the process.
- A kernel of a lazily compiled library is given at once while another library restored from the proxy cache is being adapted.
- A write to the proxy cache that fails leaves no temporary file; temporary files left more than ten minutes before are removed at the first use of the cache, and one left by an earlier start of a container no longer makes a later write fail.
- An unload of a module waits for a launch from its second translation and for a setting given to it, instead of possibly letting them reach a function of another module.
- `cuModuleLoad` and `cuLibraryLoadFromFile` adapt a file whose path is not valid UTF-8.
- A fat binary whose compressed entry announces an uncompressed size that no entry has (above 2 GB, or more than LZ4 can make of its bytes) is refused with an error instead of ending the program on a failed allocation; a zstd entry is no longer decoded past the size it announces. The proxy's cache reader no longer reserves more than its files can hold.
- libnvJitLink.so.13: `nvJitLinkCreate` with a null array of options, or a null option, returns `NVJITLINK_ERROR_INVALID_INPUT` instead of crashing; `nvJitLinkAddData` of no bytes or at a null address returns `NVJITLINK_ERROR_INVALID_INPUT` for every input type, as nvJitLink 12.9.2 does.
- `redux.sync` forms that the PTX ISA does not have (a sum of floats, bitwise operations on other types than b32, ...) make their kernel refused instead of being lowered; over a partial warp every lane reduces in the same order.
- With `VF_PROFILE`, each kernel keeps the counts of its first 64 distinct grid, block and dynamic shared memory sizes: a long-running program no longer makes the profile grow with every new launch shape.
- `vf ptxas`: ends with status 255 on the errors it finds itself, as ptxas; the assembler's messages name the user's input files instead of `/proc/self/fd/N`; `--apply-controls`, `--enable-func-clone-sc` and `--sanitize` take their value, and `-O` no longer takes the argument after it; the runfiles views are named by a fixed hash, and the ones nothing will use again are removed.
- The launcher no longer takes a leftover of an interrupted NCCL download (`libnccl.so.2.vf-part`), or an empty `cuda-12.9.2/lib`, for the CUDA 12.9.2 libraries, which hid a toolkit found next; `vf doctor` says which CUDA 12.9.2 directory it took and why.
- `vf fetch` unpacks each archive and the NCCL runtime in a staging directory and moves them into place: a fetch that fails or is stopped leaves the previous files rather than a mix, SIGTERM and SIGHUP stop it as Ctrl-C does, and the next fetch removes what an interrupted one left.
- Run under itself (`vf vf ...`, or after `eval "$(vf env)"`), the launcher no longer adds its `VF_AUDIT_MAP` entries again.
- `install.sh` removes the staging directories that killed installers left more than a day before.

### Changed

- A value of `VF_CAPABILITY` that is not a capability (`8`, `sm_80`) or is below 7.0 is said, once, and not used.
- The launcher keeps the audit libraries already in `LD_AUDIT`, after its own and without its own twice; `vf doctor` shows the whole value.
- The launcher refuses a CUDA 12.9.2 directory (`--cuda12`, `VF_CUDA12`) or an installation whose path has a `:`, which `LD_LIBRARY_PATH` and `VF_AUDIT_MAP` would split.
- `install.sh` holds its downloads to HTTPS, redirections included; `VF_INSTALL_BASE` must be `https://` or `file://` (`http://` with `VF_INSTALL_ALLOW_HTTP=1`).
- `scripts/fetch-redist.py` refuses a manifest given by an `http://` URL unless it is given `--allow-http`.
- The documents say what 0.1.3 does: the translation of sm_80 machine code is not checked by the independent validator (`docs/validation.md`, section 8); `LIMITS.md` says what `VF_CAPABILITY` does not cover; `INSTALL.md` lists every variable that the layer, the launcher and the installer read.

### Added

- `VF_CAPABILITY_ALL=1` tells the callers that are held to 7.5 what `VF_CAPABILITY` asks all the same, for a program that does not launch the kernels it would choose for it (the SGLang guide sets it).
- When the layer refuses a kernel of NCCL, it says once, at the default log level, that NCCL's CUDA 12 build is needed and that `vf fetch --only libnccl` downloads it. `LIMITS.md` and the SGLang guide say that any use of several GPUs from PyTorch (`torch.distributed`, `DataParallel`) needs it.
- `vf translate IN OUT MAP` writes `MAP.lives` beside the map, as `reference/translate.py` does, so that `reference/validate.py` can check a translation made by the layer's translator.
- With an environment made by `vf venv` activated, `vf version` and `vf doctor` say which release it runs when it is not theirs.

### Tests

- `torch-ops.py` and `torch-models.py` read verdicts from a channel of their own, so kernel output can no longer hide or forge one. A case that does not end within `VF_CASE_WAIT` (600 s) is stopped and counted as a CRASH. A missing or unknown verdict, or a short count, fails the run. New host test `tests/gpu/torch-workers.py`, run in CI.
- Validator: rule V now follows the moves of uniform predicates kept in the frame (`mem: load/store UP<n>`). Four new controls (`up_standin_l3`, `up_wrong_word_l3`, `up_no_store_l3`, `up_unwaited_l3`): 107 controls, 104 objected to.
- `vllm.sh` stage c2 checks every sampled token against the top-k and top-p of the engine's own log-probabilities; `vllm.sh` and `sglang.sh` take a stage's verdict only from the program's `verdict:` line.
- `vllm-topk-topp-run.sh` compares vLLM's sampling kernel with the same kernel run on the CPU by Triton's interpreter (18 of 18 the same), and it and its test end with a failure when a case fails.
- `launcher.sh` compares the matmul with the CPU within 1e-3 (it was 1.0); `compare.sh` reports a reference translator job that died; `llamacpp-ops.sh` counts only the case lines; `torch-models.sh` and `torch-bench.sh` look for PyTorch under `VF_TEST_ROOT`, like the other scripts.
- New host tests: `tests/proxy/lifecycle.sh` (fork, contexts, the lazy lock, cache temporaries, file names), `link_refusals.py`, `nccl_remedy.py`; new GPU cases `refuse-link` and `ptx-reduction-exited-f32` in `tests/gpu/run.sh` (113 cases).
- The CI also runs the host tests `graph_fit.sh`, `lifecycle.sh`, `library_entries.py`, `driver_capability.py`, `nvml_capability.py`, `tests/forward/stack.sh`, `tests/launcher/install.py` and `tests/launcher/fetch-redist.py`, and names its actions by commit.

<!-- TO FILL BEFORE THE RELEASE: the lines of the fixes of the second review of this branch (findings R-1 to R-9) go into the groups above. -->

## 0.1.2 (2026-11-22)

- VF can fetch the CUDA 12 build of NCCL and select it for both SGLang and PyTorch. SGLang generated with the 35B model on two V100s over NVLink. A separate NCCL all-reduce passed on four V100s. The four-GPU SGLang server stopped during startup.
- Add an opt-in register allocation policy (`VF_BORROW=9`) for CUDA Graph kernels whose register use would otherwise prevent a legal block size on Volta.
- Extend BF16 and machine-code translation, preserve compatible indirect calls in standalone PTX, and adapt PTX passed to external assemblers through `vf ptxas`.
- Install releases atomically and share fetched CUDA libraries across installed versions.
- Add guides and measured prefill and decode rates for eight inference engines on V100 with quantized Qwen3.5-35B-A3B weights. See the inference engine table for each quantization and native comparison.

## 0.1.1 (2026-11-21)

- CUDA runtime libraries built with separate device compilation can now use their relocatable PTX on Volta. VF compiles independent kernels on demand and preserves cross-object calls when linking the complete library.
- Matching weak CUDA kernels and kernels with launch-supplied shared memory can be compiled individually across linked objects. Conflicting declarations still go through the device linker.
- The sm_75 machine-code translator handles uniform sign extension and mixed-register address calculations emitted by Triton.
- The PTX translator handles packed BF16 subtraction, packed FP8-to-BF16 conversion and BF16-to-FP16 conversion.
- vLLM's standard NVML backend now sees the same capability as its CUDA 13 PyTorch runtime under `vf`. Its NVML query reports 7.5 for Volta by default and honors `VF_CAPABILITY`.
- Under `vf` or `vf venv`, direct CUDA driver clients with no embedded GPU code also see 7.5 on a V100 by default. Direct clients can still read native Volta kernel binary versions. CUDA 12 runtimes keep their previous handling of explicit overrides, and other GPUs retain their physical capabilities.

## 0.1.0 (2026-11-21)

The first version made public. It runs software built with CUDA 13 on a Tesla V100 with the 580
driver:

- `libcuda.so.1`, in front of NVIDIA's driver library: PTX for later GPUs retargeted, with the
  instructions that Volta lacks rewritten (the tensor-core products of Turing, Ampere and
  Blackwell, matrix loads, bf16, FP8, FP4, block-scaled products); machine code for sm_75
  translated to sm_70 when it is loaded; kernels that cannot run refused with a reason; a cache
  of what was worked out.
- Forwarding libraries for 22 libraries of CUDA 13, served by their CUDA 12.9.2 builds; an
  nvJitLink 13 served by nvJitLink 12.9.2; an audit library for programs that open their
  libraries by path.
- `vf`, a launcher (`volta-forever` under its long name): `vf COMMAND` runs a command through
  the layer, `vf venv DIR` makes a virtual environment of Python that runs through it by
  itself, `env` prints the environment for a shell session, `fetch` downloads the CUDA 12.9.2
  libraries from NVIDIA's servers, `doctor` says what is missing; the offline tool is
  `vf-translate`, which the launcher has as `vf inspect`, `extract`, `adapt`, `translate`.
- A program can ask whether the layer is there: `volta_forever_version()` and
  `volta_forever_abi()` in `libcuda.so.1`, `VOLTA_FOREVER` in the launcher's environment.
- Tested with PyTorch 2.14.0, llama.cpp b12418, vLLM 0.34.0, SGLang 0.5.23 and ZML
  (`LIMITS.md`).

Known limits of this version are in `LIMITS.md`; how the translations are checked, and what no
check covers, in `docs/validation.md`.
