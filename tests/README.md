# Test suites

What is here: the tests on a GPU (`tests/gpu/`), from single kernels to whole engines; the
tests of the machine-code translator, which need no GPU (`tests/sass/`); a test of the
forwarding libraries' stack handling (`tests/forward/`); the launcher tests
(`tests/launcher/`); and the kernels, programs and reference vectors that they use
(`tests/kernels/`). `docs/validation.md` says what each of them can and cannot show.

## Settings

Every test script reads `tests/site.env` when it exists (see `tests/site.env.example`) and uses
`VF_TEST_ROOT` as the root of its working files. Each setting is an environment variable with a
default; a variable already set in the environment wins.

| Variable | Default | Used by |
|---|---|---|
| `VF_TEST_ROOT` | `$XDG_CACHE_HOME/volta-forever/tests` | all scripts |
| `VF_OUT` | `$VF_TEST_ROOT` | GPU scripts: build outputs and logs |
| `VF_PROXY` | `dist/lib` of this tree | GPU scripts: the `libcuda.so.1` to test |
| `VF_LAUNCHER` | (not set) | torch, torch-models, vLLM, SGLang: `1` takes the layer's environment from the launcher (`vf env` of this tree) instead of setting it by hand |
| `VF_JIT_CACHE` | `$VF_TEST_ROOT/jit-cache` | llama.cpp, ZML: compiled kernels |
| `VF_TORCH` | `$VF_TEST_ROOT/torch-cu130/site` | torch, vLLM, SGLang: PyTorch 2.14.0+cu130 |
| `VF_TORCH_NATIVE` | `$VF_TEST_ROOT/torch-cu126/site` | torch-bench, torch-models: PyTorch with sm_70 code |
| `VF_NATIVE_LLAMA` | `$VF_TEST_ROOT/llama-b12418-sm70/build/bin` | llamacpp-bench, llamacpp-ops: native sm_70 build |
| `VF_AUDIT_MAP_FILE` | `$VF_OUT/torch-audit.map`, written when missing | torch, vLLM, SGLang: the map that puts the forwarding libraries in front of the cuBLAS that the PyTorch wheel carries (`INSTALL.md`, way B) |
| `VF_VLLM_SITE` | `$VF_TEST_ROOT/vllm/site` | vLLM: vLLM 0.34.0 and its dependencies |
| `VF_VLLM_MODEL` | `$VF_TEST_ROOT/models/Qwen2.5-0.5B-Instruct` | vLLM: the test model |
| `VF_VLLM_VENV` | (none) | vLLM: a virtual environment, instead of the directories above |
| `VF_SGLANG_SITE` | `$VF_TEST_ROOT/sglang/site` | SGLang: SGLang 0.5.23 and its dependencies |
| `VF_SGLANG_VENV` | (none) | SGLang: a virtual environment, instead of the directories above |
| `VF_MODEL` | `$VF_TEST_ROOT/models/Qwen2.5-0.5B-Instruct` | SGLang: the test model |
| `VF_ZML` | `$VF_TEST_ROOT/zml/build` | ZML: the build of `tests/gpu/zml/build-zml.sh` |
| `VF_ZML_MODEL` | `$VF_TEST_ROOT/models/Qwen3.5-9B-AWQ-fp16-text` | ZML: the model (9 GB) |
| `VF_VLLM_EXTRA` | `$VF_TEST_ROOT/vllm-extra/site` | vLLM: packages that the sites above lack (loguru and others) |
| `VF_VLLM_DEPS` | `$VF_TEST_ROOT/vllm-deps/site` | vLLM: what the sites above lack of what vLLM declares |
| `VF_CUTLASS_PREFIX` | `$VF_TEST_ROOT/cutlass-prefix` | SGLang: the CUTLASS DSL |
| `VF_PYTHON_DEV` | `$VF_TEST_ROOT/python-dev` | vLLM, SGLang: Python 3.12 headers |
| `VF_SERVER_WAIT` | `1800` | vLLM, SGLang: seconds to wait for the HTTP server to answer |
| `VF_ENGINE_WAIT` | `1800` | SGLang: seconds that an engine stage may take |
| `VF_CASE_WAIT` | `600` | torch, torch-models: seconds that a case may take before it is stopped and counted as a CRASH |
| `TORCH` | `$VF_TEST_ROOT/torch-cubins` | SASS: PyTorch cubins for the corpus |

A setting of one script alone is described in the header of that script (for example
`VF_ZML_CUDNN` of `zml.sh`, `VF_BENCH` of `llamacpp-bench.sh`, `VF_TOOL` and `JOBS` of the
scripts of `tests/sass/`).

## What PASS means

- `run.sh`: the test kernels, loaded as PTX and as machine code through the proxy (retargeting,
  translation, refusal, the cache, the second translation at launch, the CUDA runtime), give
  what the native sm_70 build of the same source, the host models or the reference vectors of
  an RTX 5090 say. 126 cases at 0.1.3, including nonconsecutive MMA fragments, a partial warp
  reduction, a zero-filled asynchronous copy, sm_80 machine code translated, refused lookups
  (error 218 and a NULL handle, through a module, a library and the driver's linker), a program
  with a refused kernel under the CUDA 12 and 13 runtimes in both loading modes
  (`CUDA_MODULE_LOADING`), and lookup tables asked for by name, answered as NVIDIA's driver
  answers them, except where the layer refuses one (`tables.py`).
- `forward.sh`: for each of ten libraries, a program built against CUDA 13.5 calls a few
  functions through the forwarding library and gets the host's result, five of them again under
  `CUDA_MODULE_LOADING=EAGER` (cuBLAS is exercised by `run.sh`, under both loading modes).
- `forward-bf16.sh`: the opt-in bf16 cuBLAS variants on a V100, with a test build of their
  runtime: the skinny path's results bit for bit against cuBLAS 12.9 where it serves, and the
  inputs that tell a fused multiply-add chain, a multiply-and-add chain and the epilogue's orders
  apart (`bf16-arith.c`); the tensor-core path within 32 units of 2^-24 sum |a b| of fp64,
  deterministic; every forwarded call with 12.9's bits; what a program's graphs, capture modes,
  threads and contexts see (`bf16-graphs.c`, `bf16-threads.c`, `bf16-life.c`, `bf16-reserve.c`;
  a graph captured through the variant behaves step for step as with 12.9 alone and replays 12.9's
  bits, with tensor-core-sized calls, forwarded, and with skinny ones, `bf16-graphs-skinny`, served
  by kernel nodes of the runtime's module, which replay after every handle is destroyed);
  cuBLASLt calls that keep 12.9's behaviour: an explicit algorithm other than 12.9's heuristic one,
  and a workspace size without a workspace (`bf16-lt.c`); the tensor-core path's fallback in the TN
  layout of linear layers, bit for bit against a host model (`bf16-fallback.c`); the runtime told
  the real capability under the layer's settings, under both module loading modes
  (`CUDA_MODULE_LOADING=EAGER` and `LAZY`);
  the checked-in fat binary rebuilt when the CUDA 12.9.2 compiler is there (`bf16-kernels`); with
  `VF_BF16_SANITIZE` set, under compute-sanitizer. `bf16-skinny.c` is the measurement behind the
  skinny path's rule and bounds (12.9's kernel per configuration; path, bits over the m rows of each
  column, writes outside them, error and time per shape), and `bf16-cost.c` the one behind the
  documented host times, fallback cost and same-sign error; both run by hand.
- `tests/forward/bf16.sh` (no GPU, in the CI): the variants generated from stand-in libraries, and
  their runtime against stand-ins for the driver and cuBLAS that record every call (H1-H7, H9 of
  the design review), each control making its test fail.
- `torch.sh`: every PyTorch operation on the GPU, computed by translated kernels, gives what
  the CPU gives (within the tolerance of different rounding).
- `llamacpp.sh`: the next-token distributions of a text are those of the CPU's (mean KL
  divergence at most 0.005, same most likely token at least 95 %).
- `llamacpp-bench.sh`, `llamacpp-ops.sh`: speed and correctness against a native sm_70 build.
- `vllm.sh`, `sglang.sh`: the tokens generated by the engine are those of the CPU's (plain
  greedy), or the layer says which kernel it cannot give; in stage c2 of `vllm.sh`, every token
  sampled is among the top k and within the top p of the engine's own log-probabilities. A
  stage's verdict is the last `verdict:` line that its program prints, and counts only when the
  program then ends by itself, with status 0, within its time (`tests/gpu/verdict.sh`).
- `zml.sh`: the program answers with the right word and reports a speed.
- `triton.sh`: each of the 180 cases of the quick subset of the Triton corpus gives on the V100
  what the same kernel gives on the CPU under Triton's interpreter (or the case's PyTorch
  computation), bit for bit or within the tolerance the case states; its five runs give the same
  bits; and the layer's two routes, the translated sm_75 machine code and the retargeted PTX of
  the same kernel, give the same bits. All of them on the same inputs: made once by the step that
  makes the references and saved with them, and named by their digest in the report. A case
  without a reference (the interpreter cannot run it) is compared between its runs and the routes
  only (`SKIP-REF`); one whose machine code the layer refuses and runs as the PTX kept in the
  cubin is `PASS-PTX`.
- `compare.sh`: the Rust translator and the Python reference give the same bytes, the same
  instruction map, the same `MAP.lives` and the same refusals for every cubin, sm_80 test
  cubins included (at 0.1.3: 489 of 489, the 14 test cubins and PyTorch's 475).
- `validate-corpus.sh`: the independent validator finds 0 problems over the corpus (of sm_75
  code: the validator does not read sm_80 code, `docs/validation.md`, section 8).
- `controls.sh`: the validator objects to translations that are made wrong on purpose (the
  controls of `reference/controls.py`, each in a process of its own), in every cubin whose
  translation a control changes but the cases looked at one by one (at 0.1.3: 108 controls, 105
  objected to; `docs/validation.md`, section 3, says what the other three are).
- `heldout-validate.sh`: the same, on a corpus kept apart from the one the rules were found on.
- `fuzz.py`: on damaged cubins, the two translators agree on what they make of every mutant.

## First run

The first run of a test that launches CUDA kernels through the proxy compiles them (the
driver's PTX compiler turns the retargeted PTX into machine code for the GPU). This takes
minutes (llama.cpp: 3 to 5 minutes for the small model; vLLM and SGLang: longer). The compiled
code is kept in `$VF_JIT_CACHE` and reused on later runs.

## Suites

### GPU tests (`tests/gpu/`)

Need a Tesla V100, the layer built (`scripts/build.sh`), the forwarding libraries built
(`scripts/build-forwarders.sh`), and the CUDA toolkits fetched (`scripts/fetch-all.sh`).

| Script | What it runs | Compared with | Time | What it needs |
|---|---|---|---|---|
| `run.sh` | test kernels through the proxy | native sm_70 builds, host models, RTX 5090 reference vectors | a minute or two | toolkits |
| `forward.sh` | forwarding libraries | host computation | a minute | toolkits |
| `torch.sh` | 925 PyTorch operations | the CPU | 3 min | PyTorch 2.14.0+cu130 wheel |
| `torch-models.sh` | 32 small whole models | the CPU | 4 min | PyTorch 2.14.0+cu130 wheel |
| `torch-bench.sh` | 89 PyTorch operations (speed) | PyTorch with sm_70 code | 12 min | both PyTorch wheels |
| `llamacpp.sh` | llama.cpp, official build | the CPU | 2 min | `scripts/fetch-all.sh demos` |
| `llamacpp-bench.sh` | llama.cpp (speed) | native sm_70 build | 10 min | demos, native build |
| `llamacpp-ops.sh` | llama.cpp's backend test | native sm_70 build (`perf`) or the CPU (`test`) | 6 to 20 min | demos, native build, sources |
| `llamacpp-nvfp4.sh` | llama.cpp with NVFP4 weights | the CPU, native build | depends on the model | demos, native build, an NVFP4 GGUF model |
| `vllm.sh` | vLLM's engine and server | the CPU (greedy) | 10 min | vLLM 0.34.0, PyTorch, model, Python headers |
| `sglang.sh` | SGLang's engine and server | the CPU (greedy) | 10 min | SGLang 0.5.23, PyTorch, model, Python headers |
| `zml.sh` | ZML's LLM example | the word "Paris"; reports a speed | 5 min and the time to read the model twice | ZML build, a 9B model |
| `profile-torch.sh` | PyTorch with the layer's profile | the profile has the right columns | 15 min | both PyTorch wheels |
| `launcher.sh` | PyTorch and llama.cpp through the launcher (`vf`), and a virtual environment made by `vf venv` whose Python is started with no prefix | a matmul against the CPU (within 1e-3); the forwarding cuBLAS in `/proc/self/maps`; text from llama.cpp; the layer says that it is there, by its variable and by the functions of `libcuda.so.1` | 10 min | PyTorch 2.14.0+cu130 wheel, demos |
| `vllm-topk-topp-run.sh` | vLLM's top-k and top-p sampling kernel (Triton), 18 cases, in two passes, each with a proxy cache of its own: at `VF_BORROW=1` (levels 2, 1, 0), then at `VF_BORROW=5` (level 3 with room made, as the second translation at launch translates) | the same kernel run on the CPU by Triton's interpreter | not timed | vLLM, PyTorch, Python headers; the GPU chosen by `CUDA_VISIBLE_DEVICES` |
| `triton.sh` | 180 Triton kernels and configurations (the quick subset of `triton_corpus.py`, run by `triton-diff.py`): element-wise operations in every type, conversions, reductions, scans, sorts, `tl.dot` in several types, precisions, tiles, warps and stages, attention, gathers, atomics, loops, grids, libdevice; each through the translated machine code (5 runs) and through the retargeted PTX of the same kernel (2 runs) | Triton's interpreter on the CPU, or a PyTorch computation on the CPU; the runs and the two routes with each other | 4 min (2 of them for the interpreter) | PyTorch 2.14.0+cu130 with its Triton, numpy (`VF_TRITON_EXTRA`), Python headers; or `VF_TRITON_PYTHON` |
| `sglang-decode-graph.py` | SGLang's FP16 decode attention captured in CUDA graphs and replayed with new inputs, run through the layer with SGLang and PyTorch on `PYTHONPATH` | the CPU in FP32 | not timed | SGLang, PyTorch |

(Times on one V100 with everything already compiled and in the file cache.)

The whole Triton corpus (3,516 cases; `triton_corpus.py` says how cases are made and checked) is
run by hand, the reference on any machine and the GPU part on the V100, in as many processes as
wanted (`--shard I/N`; a case that stops its process is counted as a CRASH and the next process
goes on from the case after it):

```sh
TRITON_INTERPRET=1 python3 tests/gpu/triton-diff.py interp --level full --out REF --jobs 24
python3 tests/gpu/triton-diff.py gpu --route sass --ref REF --out sass.jsonl --level full --runs 5
python3 tests/gpu/triton-diff.py gpu --route ptx --ref REF --out ptx.jsonl --level full --runs 5
python3 tests/gpu/triton-diff.py report --level full sass.jsonl ptx.jsonl
```

(the `gpu` commands through the layer, with `VF_LOG_LEVEL=2` and `VF_LOG` set, so that each case
says which route its modules took; `interp` saves each case's inputs in `REF/inputs`, made in one
thread, and the `gpu` commands take them from there, so that a reference made on another machine
was computed on the inputs the GPU is given (made again, PyTorch's tanh, which is MKL's, does not
give the same last bits on every machine, nor in every process: `ld-erfinv-f64` once had other
inputs in one of the two processes of the routes, a block of 2,048 values that one thread computed
with MKL's AVX2 code of low accuracy); `triton-diff.py inputs --out DIR` makes the inputs alone and
prints the digest of all of them; `--vendor` adds the kernels of vLLM and SGLang of
`triton_vendor.py`, run with the Python of each engine's environment: 151 cases with vLLM, 90 with
SGLang; Triton builds its C launcher on the node, so `CPATH` must name Python 3.12 headers there).

`torch-ops.py` and `torch-models.py` run their cases in worker processes, which write their
verdicts on a descriptor or to a file of their own (what kernels print on standard output cannot
hide one); a case that has not ended after `$VF_CASE_WAIT` seconds (default 600) is stopped and
counted as a CRASH. `python3 tests/gpu/torch-workers.py` checks that on the host, with fake
workers (no GPU, no PyTorch, 15 seconds): kernels that print, a worker that fills its standard
error or does not end, a line that is not a verdict, fewer verdicts than cases, a second verdict
for a case, a verdict for a case that the worker did not start or did not name.
`python3 tests/gpu/vllm-engine-verdicts.py` checks, with a stand-in for vLLM (no GPU, no vLLM),
the verdict lines of `vllm-engine.py` (stages c and c2 of `vllm.sh`): fewer results than prompts,
or none, or a reference with fewer texts, do not pass, nor a sampled token whose log-probability is
NaN or above 0; and the verdict of a stage of `vllm.sh` and `sglang.sh` (`tests/gpu/verdict.sh`): a
program that prints its verdict and then fails, or does not end within its time, does not pass.

### Launcher tests (`tests/launcher/`)

No GPU needed. Exercises the launcher (`vf`) on fake installations (both the archive and
source-tree layouts): environment variables, path order, map entries, user priority, options,
what is done when only some of the CUDA 12.9.2 libraries are there (a forwarding library with
nothing to forward to is not used), argument passing, exit status, symlinks, its long name,
`doctor`, `fetch` command line, the
subcommands of the offline tool, help and error cases, the bf16 cuBLAS variants that
`VF_CUBLAS_BF16_TC` and `VF_CUBLAS_BF16_SKINNY` select (both names mapped or none, what is
refused, what `doctor` says); and `vf venv` with the Python of the
machine: an environment whose Python runs through the launcher with no prefix (arguments,
exit status, standard input, a child process), what it refuses, how it is turned off.

| Script | What it does | Time | What it needs |
|---|---|---|---|
| `launcher.sh` | 108 checks; run it under `dash` and under `bash --posix` | seconds | the launcher (`scripts/vf`); python3 3.10 or later with `venv` for the checks of `vf venv` |
| `install.py` | `install.sh` on local releases: an upgrade that keeps the CUDA 12.9.2 libraries and the `vf venv` environments of the release before, installs stopped half-way, two at once, what killed ones left, plain HTTP refused, redirections from HTTPS (through curl and through wget); under `dash`, or `bash --posix` with `VF_TEST_SHELL=bash` | seconds | dash, flock, python3 3.10 or later with `venv`; openssl for the redirections (skipped without it) |
| `fetch-redist.py` | `scripts/fetch-redist.py` on local archives: retries after an interruption, concurrent fetches, runtime-only installs, the NCCL wheel, a failure or a signal half-way, a component killed while it is moved into place (the journal, which the next fetch completes), what an interrupted fetch left, a manifest over plain HTTP | seconds | python3, tar |

Run the last two with `python3 tests/launcher/install.py` and `python3 tests/launcher/fetch-redist.py`.

### Tests of the machine-code translator (`tests/sass/`)

No GPU needed. They need the reference translator (`reference/`), the instruction descriptions
that it reads (fetched by `scripts/fetch-all.sh denvdis`), NVIDIA's disassembler for the
validator (in `toolkits/`), and the `vf` tool built (`cargo build --release -p vf-cli`).

| Script | What it does | Time | What it needs |
|---|---|---|---|
| `compare.sh` | the Rust translator against the Python reference: bytes, maps, lives, refusals | seconds for the test kernels; a minute for PyTorch's 475 cubins with 48 processes | `reference/`, `vf` tool, cubins |
| `validate-corpus.sh` | the independent validator over a corpus | a quarter of an hour for PyTorch's 475 cubins with 48 processes | `reference/`, `toolkits/`, cubins |
| `controls.sh` | every control of the validator, in five passes | 40 minutes with 64 processes for the cubins it names | `reference/`, `toolkits/`, cubins |
| `heldout-extract.sh` | extract the cubins of other libraries | minutes | `vf` tool, the libraries |
| `heldout-validate.sh` | translate and validate the held-out corpus, keeping why each function is refused | an hour with 40 processes | `reference/`, `toolkits/`, the corpus |
| `fuzz.py` | damaged cubins: the two translators must agree | two and a half minutes for 20,000 mutants with 20 processes | `reference/`, `vf` tool, small cubins |

### The second translation, by every way to it (`tests/proxy/`)

No GPU needed. A stand-in for NVIDIA's driver library, built by the test from C, refuses the
launches of the first image that is loaded; the proxy translates the test cubin and loads it
through the stand-in.

| Script | What it does | Time | What it needs |
|---|---|---|---|
| `fit_api.sh` | The second translation at launch by every way a program can come to it: a variable or a surface reference asked for by name (both exported names, and what `cuGetProcAddress` gives) excludes it; the launch entries for the per-thread default stream, a library's kernels taken by enumeration, and the functions of the module that the driver gives for a library are launched from it; a library and that module are one image; an enumeration into an array larger than the library's number of kernels reads only what the driver wrote; what is set on a library's kernel is given to the function of the same name in the library's module, for the device of its context, unless the function has a setting of its own, and when it is set later too; a launch whose second translation was being made when the program asked for data is refused; two settings of one kernel made at once leave it and its second image with the same value; an unload of the module waits for the settings given to its second image, for the lookup of its kernel there and for the launch from there; a second translation made for a module that goes meanwhile is not taken for the module loaded next with the same handle, and is unloaded when the program unloaded the module, not when its context was destroyed. 46 checks | seconds | the proxy built, `tests/kernels/k_fit_test.135.sm75.cubin`, gcc |
| `tests/proxy/graph_fit.sh [PROXY_LIBRARY]` | CUDA graph register pressure with a stand-in driver reading the actual translated ELF. Eight recording, update and capture paths under the usual, bounded and block-size-aware policies | seconds | the proxy, the same cubin and a C compiler |
| `marker.sh` | The functions by which a program can tell that the layer is there (`volta_forever_abi`, `volta_forever_version`): found by `dlsym` in the layer's `libcuda.so.1`, with the interface number 1 and the workspace's version | seconds | the proxy built, gcc |
| `lifecycle.sh` | What the layer keeps over the life of a process, and what it does at its end: a child made by `fork()` while another thread of the parent waits in a write of the log ends when it calls `exit()`; what the layer keeps of the modules of a context (adapted images, kernels, originals kept for a second translation, second translations) is dropped when the context is destroyed, released for the last time or reset, by both names of each call, and not otherwise (a release that is not the last, another context, a library, a module that another thread loads once the driver destroyed the context, in a context made at its address or with the handle of one of its modules, and a retain of the primary context by another thread meanwhile); while a library restored from the cache is adapted for a kernel whose code the cache does not have, a kernel of another library is given at once, the same kernel asked for by another thread is the same, and another kernel of the library asked for meanwhile is given from the same adaptation; a write to the cache that fails leaves no temporary file, and the first use of the cache removes one left an hour before; a file whose name is not UTF-8 is adapted as any other. Each case in a process of its own | seconds | the proxy built, gcc, `tests/kernels/k_fit_test.135.sm75.cubin` |

Eight more host regressions exercise the built libraries directly:

```sh
python3 tests/proxy/library_entries.py dist/lib/libcuda.so.1
python3 tests/proxy/driver_capability.py dist/lib/libcuda.so.1
python3 tests/proxy/nvml_capability.py dist/lib/libvf_audit.so
python3 tests/proxy/link_refusals.py dist/lib/libcuda.so.1
python3 tests/proxy/nccl_remedy.py dist/lib/libcuda.so.1
python3 tests/proxy/jitlink_objects.py dist/lib/libnvJitLink.so.13 toolkits/cuda-12.9.2
python3 tests/proxy/jitlink_fork.py dist/lib/libnvJitLink.so.13
python3 tests/proxy/readonly_tables.py dist/lib/libcuda.so.1
```

The first checks filtered kernel counts and enumeration with a stand-in driver, including
deferred loads and the lifetime of replacement images, and that a lookup that fails leaves NULL
in the handle (refused by the layer, by the driver's compiler when a deferred library is loaded
whole for it, or of a name the library does not have). The second checks the compute capability that the proxy gives (device attributes,
`cuDeviceComputeCapability`, the attributes of a function and of a kernel) to callers whose
code is for different GPUs, on a stand-in driver with a V100 and three later cards, with and
without the launcher's marker and `VF_CAPABILITY`; it needs gcc and objcopy. The third checks
the capability that NVML gives through the audit library, by `ctypes` (as `pynvml` asks) and
to a library linked against it; it needs gcc. The fourth checks, with another stand-in, that
a kernel refused in an input of the driver's linker is refused in the image the link makes,
and in a copy of it, that a kernel that the note of a cubin names (as the layer's nvJitLink
writes it) is refused wherever the cubin is loaded (each loading function, a file, a fat binary,
a wrapper, an input of the driver's linker), and that what is known of a link or a module made
with a handle that the driver gives again, while the destruction or unload of the one before
returns, is kept. The fifth
checks that a refused kernel of NCCL makes the layer say once, at the default log level, that
NCCL's CUDA 12 build is the remedy. The sixth links a native sm_70 device function from both a
host object and a static library with the CUDA 12.9 toolkit, reads a file whose name is not
UTF-8 by that name (not the one its text names), and checks that PTX in which every kernel
traps is refused as text and in a fat binary alike (and Turing machine code whose every kernel
the translator refuses), and that with one kernel that runs both are linked, into a cubin that
names the kernel that traps in a note, which a link of that cubin keeps. The seventh
checks, with a stand-in for nvJitLink 12.9.2, that a child made by `fork()` while another
thread of the parent waits in a write of the adapter's log ends when it calls `exit()`; it
needs a C compiler. The eighth checks, with another stand-in that keeps the text of what it is
given, that a deferred library gives a kernel that only reads a table of its module from a
sub-module that holds a copy of the table, and then refuses the table by every name the program
could ask for it with (error 801; a name that the module does not have stays 500), while a
variable that kernels write is given; and that a table
the program was given first keeps the kernels that read it in the whole module, also one that
another thread asks for while the whole module is compiled for the table (the stand-in holds that
compilation). Each case in a
process of its own, a second time from the proxy cache; it needs a C compiler.

### Forwarding stack test (`tests/forward/`)

No GPU needed. Builds and tests the status-renumbering trampolines of the forwarding
libraries against a mock library.

| Script | What it does | Time | What it needs |
|---|---|---|---|
| `stack.sh` | stack-word copying correctness | seconds | Cargo (`vf-fwdgen` crate) |
| `package-refusals.sh` | what `scripts/package.sh` refuses to pack: a built tree with a forwarding library missing, with one older than the sources, with the report of a run for one library only, with some but not all of the three files of the bf16 cuBLAS variants (`forward/bf16tc`); and that it packs the three in `forward/bf16tc` when they are all there | seconds | a built `dist/` (`scripts/build.sh`, `scripts/build-forwarders.sh`) |

### What the CI runs

On every push and pull request, `.github/workflows/ci.yml` runs the formatting and lint
checks and the unit tests (`cargo test`, 422 at 0.1.3); `scripts/build.sh`; `fit_api.sh`, `marker.sh`,
`graph_fit.sh`, `lifecycle.sh`, `library_entries.py`, `readonly_tables.py`, `driver_capability.py`,
`link_refusals.py`, `nccl_remedy.py`, `jitlink_fork.py` and `nvml_capability.py` of
`tests/proxy/`; `tests/forward/stack.sh`; `launcher.sh` under `dash` and under `bash --posix`;
`tests/gpu/torch-workers.py` and `tests/gpu/vllm-engine-verdicts.py`; `install.py` under `dash`
and with `VF_TEST_SHELL=bash`, and `fetch-redist.py`; `tests/sass/compare.sh` on the test
kernels; then `install.sh` on five distributions. It does not run `jitlink_objects.py`, which
needs the CUDA 12.9.2 toolkit, nor
`tests/forward/package-refusals.sh`, which needs the forwarding libraries built from NVIDIA's
toolkits (`scripts/fetch-all.sh redist`, `scripts/build-forwarders.sh`): it fetches neither.
The tests on a GPU and the SASS tests over whole corpora are run by hand.

## Preparing the dependencies

### Toolkits and test models

```sh
scripts/fetch-all.sh redist cudnn   # CUDA 12.9.2 and 13.5.0, cuDNN, from NVIDIA's servers (tens of GB)
scripts/fetch-all.sh demos          # the official llama.cpp build, two small models
scripts/fetch-all.sh denvdis        # the instruction descriptions, for reference/
scripts/fetch-all.sh src            # sources that some tests build (llama.cpp's backend test)
```

### PyTorch 2.14.0+cu130 (stock wheel, sm_75 and later)

```sh
python3 -m pip install --target $VF_TEST_ROOT/torch-cu130/site \
    --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple torch==2.14.0
```

### PyTorch 2.14.0+cu126 (native sm_70 code, for benchmarks)

```sh
python3 -m pip install --target $VF_TEST_ROOT/torch-cu126/site \
    --index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.org/simple torch==2.14.0
```

### Native llama.cpp build for sm_70

A build of the same llama.cpp sources for sm_70 with CUDA 12.9.2, for speed comparisons.

```sh
cmake -S src/llama.cpp-b12418 -B $VF_TEST_ROOT/llama-b12418-sm70/build -G Ninja \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCUDAToolkit_ROOT=toolkits/cuda-12.9.2 \
    -DCMAKE_CUDA_COMPILER=toolkits/cuda-12.9.2/bin/nvcc \
    -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=ON \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_OPENSSL=OFF \
    -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined
ninja -C $VF_TEST_ROOT/llama-b12418-sm70/build llama-completion llama-bench llama-perplexity
```

Run it with `LD_LIBRARY_PATH=$VF_TEST_ROOT/llama-b12418-sm70/build/bin:toolkits/cuda-12.9.2/lib`.

### vLLM

The simplest way:

```sh
python3 -m venv $VF_TEST_ROOT/vllm-venv
$VF_TEST_ROOT/vllm-venv/bin/pip install vllm==0.34.0
```

Then set `VF_VLLM_VENV=$VF_TEST_ROOT/vllm-venv`.

### SGLang

```sh
python3 -m venv $VF_TEST_ROOT/sglang-venv
$VF_TEST_ROOT/sglang-venv/bin/pip install sglang==0.5.23
```

Then set `VF_SGLANG_VENV=$VF_TEST_ROOT/sglang-venv`.

### ZML

Build with `tests/gpu/zml/build-zml.sh` and set `VF_ZML` to the build directory. The model
(`VF_ZML_MODEL`) is a 9 GB checkpoint that must be provided; the script stops and says what
to set when it is not found.

### PyTorch cubins (for `tests/sass/`)

Extract the cubins from the PyTorch wheel:

```sh
vf extract $VF_TEST_ROOT/torch-cu130/site/torch/lib/libtorch_cuda.so \
    $VF_TEST_ROOT/torch-cubins 75
```

### Instruction descriptions (for `tests/sass/`)

```sh
scripts/fetch-all.sh denvdis
```

## Files outside `tests/` that the suites depend on

- `scripts/build.sh`, `scripts/build-forwarders.sh`: build the layer and the forwarders
- `scripts/fetch-all.sh`: fetch the CUDA toolkits, test models and instruction descriptions
- `reference/`: the Python reference translator, validator and batch translator
- `toolkits/`: the CUDA 12.9.2 and 13.5.0 redistributable libraries (fetched)
- `demos/`: the official llama.cpp build and test models (fetched)
- `src/`: llama.cpp's sources for `llamacpp-ops.sh`, the instruction descriptions for `reference/` (fetched)
- `dist/`: the built layer (`dist/lib`) and forwarding libraries (`dist/forward`)
- `crates/vf-fatbin/tests/data/`: fat binary test files used by `run.sh`
- `tests/gpu/zml/build-zml.sh`: builds ZML for `zml.sh`
