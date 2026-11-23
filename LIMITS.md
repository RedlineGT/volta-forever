# volta-forever: known limits

What does not work through the layer, what is slower, and what was not tried. Everything was
measured on one Tesla V100 with driver 580.178.04 and no switch set; a figure carries its date
where it is not from the build of this release. `docs/` says how each figure was obtained:
`docs/validation.md` for what was compared with what, `docs/performance.md` for the timings,
`docs/emulations.md` for what is computed in place of the instructions that a V100 lacks.

## The limits of 0.1.4 to know first

- **Triton's `tl.dot` uses no tensor cores at the default capability.** Told 7.5, which is what
  Triton is told by default in a program with PyTorch, Triton 3.8 multiplies with float32 `fma`:
  its float16 and bf16 matrix products ran 3 to 150 times slower than the same kernels told 8.0
  ([Speed](#speed)). In a program with PyTorch, `VF_CAPABILITY=8.0 VF_CAPABILITY_ALL=1` is the
  only way to the tensor cores in this release, and PyTorch's own attention is then refused
  ([The capability told with `VF_CAPABILITY`](#the-capability-told-with-vf_capability)).
- **Triton's machine code is validated on a corpus, not kernel by kernel.** Kernels that Triton
  makes while a program runs, and that run from their sm_75 machine code (the default for a
  program told 7.5), were run on a V100 in 3,516 configurations of 20 kernels and 241 cases of
  vLLM's and SGLang's kernels (`tests/gpu/triton.sh` runs 180 of them): every one that Triton
  compiles gives what Triton's interpreter computes where it computes it (bit for bit, or within
  the error the case states) and what the PTX of the same kernel gives, and the independent
  validator finds no problem in their 4,308 cubins nor in the 334 of this site's runs
  (2026-11-23, with the translator and the rules of this release). A translation that stored
  wrong values in Triton's fp32 `tl.dot` kernels was fixed in 0.1.3. The kernels that Triton
  makes for other programs, shapes and settings were not looked at (`docs/validation.md`,
  sections 4, 6 and 8).
- **The translation of sm_80 machine code is not independently validated**: the validator reads
  sm_75 code only ("Cubins for sm_80 and later", below; `docs/validation.md`, section 8).
- **`VF_CAPABILITY_ALL=1` no longer reaches the libraries that ask for themselves.** It tells the
  value asked for to a CUDA runtime and to the code that uses it (PyTorch's library) only; cuDNN,
  cuFFT and cuSOLVER, which ask the driver without that runtime, are told what their own code
  allows, 7.5 for code that runs from sm_75 machine code ([The capability told with
  `VF_CAPABILITY`](#the-capability-told-with-vf_capability)).
- **A refused kernel that a program obtains without a lookup can still be launched**, and then
  traps or stops the GPU: the layer refuses a kernel when the program looks it up
  ([Kernels refused](#kernels-refused)), and has no guard at its launch in this release. Told
  8.0, cuFFT obtained kernels that exist only as sm_80 machine code, refused, through its own
  runtime, and launched them: the GPU stopped. At the settings measured it no longer does, since
  cuFFT is now told 7.5.
- **The bf16 cuBLAS variants are opt-in, and not finished.** `VF_CUBLAS_BF16_SKINNY=1` and
  `VF_CUBLAS_BF16_TC=1` are off by default. The tensor-core path's results are not cuBLAS 12.9's
  bits, and it makes bf16 training slower; there is no safe mode and no gate on k yet ([The bf16
  cuBLAS variants](#the-bf16-cublas-variants)).
- **Under eager loading (`CUDA_MODULE_LOADING=EAGER`), PyTorch takes very long to start.**
  cuBLAS, cuBLASLt, cuSPARSE, cuSOLVER, cuFFT, cuRAND and cuDNN 9.10 start under it, as on
  NVIDIA's driver, but every image that a runtime registered is adapted and compiled at the start:
  with the layer's cache off, cuBLAS 12.9 started in 30 s (2 s without the layer), and the start
  of PyTorch 2.14 had not ended after ten minutes; a later start, from the caches, was not
  measured ([Kernels refused](#kernels-refused), "Eager loading").
- **A lookup table asked for by name can be refused** (error 801), where NVIDIA's driver gives
  it, once a kernel that the layer compiled alone holds a copy of it ([Kernels
  refused](#kernels-refused), "A variable asked for by name can be refused too").
- **A cubin that the layer's nvJitLink links, loaded without the layer, keeps its kernels that
  trap**: the warning that nvJitLink writes when it links them is all there is. Likewise, the
  image that the driver's linker makes, saved and loaded in another process, gives its kernels
  that trap to the program ([Kernels refused](#kernels-refused)).
- **A kernel of one linker input that calls a function of another input that cannot run still
  traps** when it is launched: through the driver's linker and through the layer's nvJitLink,
  what traps is decided input by input ([Kernels refused](#kernels-refused)).
- **NVML reports the capability asked for** with `VF_CAPABILITY` to every caller, while a caller
  whose code runs from sm_75 machine code is told 7.5 by the driver: a program that reads both
  (vLLM does) gets two answers ([The capability told with
  `VF_CAPABILITY`](#the-capability-told-with-vf_capability)).

## Programs tested

| Program | Version | Status | Test |
|---|---|---|---|
| PyTorch | 2.14.0+cu130 and 2.14.1+cu130 (stock wheels: machine code for sm_75 and later, no PTX) | works, with the limits below | `tests/gpu/torch.sh`, `torch-models.sh`, `torch-bench.sh` |
| llama.cpp | b12418 (official build for CUDA 13.5) | works, with flash attention on and off, with quantized and NVFP4 weights | `tests/gpu/llamacpp.sh`, `llamacpp-ops.sh`, `llamacpp-nvfp4.sh` |
| ZML | upstream at `8898051c7b` with one fix of its own (`tests/gpu/zml/`) | works, told capability 7.5 and 8.0 | `tests/gpu/zml.sh` |
| vLLM | 0.34.0, installed with pip | works: the engine's greedy text is the CPU's token for token, sampling answers, the HTTP server gives the CPU's text | `tests/gpu/vllm.sh` |
| SGLang | 0.5.23, installed with pip | works, with its default attention (FlashInfer) and with `--attention-backend triton`: the engine's greedy text is the CPU's token for token | `tests/gpu/sglang.sh` |

- **vLLM** (Qwen2.5-0.5B, three prompts): every stage of the test; greedy generation at 84 to 99
  tokens a second over the runs of 2026-11-20 and 21 (83 to 97 with sampling); the HTTP server
  (the OpenAI API) with the model's repetition penalty gives the CPU's text. Its sampling kernel
  by itself against the same kernel run on the CPU by Triton's interpreter: 18 of 18 cases keep
  the same positions with the same values.
- **SGLang** (Qwen2.5-0.5B, three prompts; it makes the model float16 on a GPU that reports a
  capability under 8, whatever it is asked). With its default attention (FlashInfer): its
  engine's greedy text is the CPU's (float32) token for token, 150 of 150, at 45 to 52 tokens a
  second, one prompt at a time, and its HTTP server's too. With `--attention-backend triton`: the
  same from the engine at 55 to 59 tokens a second; the server leaves the CPU's text at one token
  of 150, where the CPU's own lead is 0.021 in logit (a near tie: `tests/gpu/greedy_compare.py`).
  Its log-probabilities differ from the CPU's by 1e-2 to 7e-2, which is what float16 does. Five
  of FlashInfer's eighteen prefill kernels are refused (below): another model or batch may ask
  for one, and the error then names it.
- **The engines were run with one small model each** and a few prompts. A program that works
  here with one model may launch, with another, a kernel that no test ran.

## Kernels refused

A kernel that the layer cannot run is refused when the program asks for it: the lookup
(`cuModuleGetFunction`, `cuLibraryGetKernel`) returns `CUDA_ERROR_INVALID_PTX` (218) and leaves
NULL in the handle, and a line on standard error names the kernel and says why. The other
kernels of its module load and run. The same holds for the image that the driver's linker makes
(`cuLinkAddData`, `cuLinkComplete`), with two exceptions: a kernel that calls a function of
another input of the link that cannot run is not refused, and traps when it is launched; and
inputs given with `cuLinkAddFile` are not adapted at all.

**Eager loading (`CUDA_MODULE_LOADING=EAGER`).** The CUDA 13 runtime asks for each kernel by
itself, and goes on without one that is refused, under eager loading as under lazy loading. The
CUDA 12 runtime, which the CUDA 12.9 libraries that the layer forwards to carry inside them,
takes the kernels of each of its fat binaries from the module of its library
(`cuLibraryGetModule`, then `cuModuleGetFunction` for each); under eager loading it does so for
every fat binary at its start, and gives up altogether at the first lookup that fails. NVIDIA's
driver gives no module for an image without code for the GPU (209,
`CUDA_ERROR_NO_BINARY_FOR_GPU`), and the runtime then does without that image. Under eager
loading the layer gives no module either (its kernels stay given by `cuLibraryGetKernel`) for an
image with a kernel that it refuses, and for an image that it made from one without code for
the V100 when the caller has code for the V100 of its own, as the CUDA 12.9 libraries do, which
choose that code here: the module would have every kernel of the image compiled at the start,
for nothing (cuBLAS 12.9 and cuBLASLt have hundreds of such images, kernels for sm_80 and later;
one took more than nine minutes to compile). So cuBLAS, cuBLASLt, cuSPARSE, cuSOLVER, cuFFT,
cuRAND and cuDNN 9.10 (its CUDA 12 build) start under eager loading as on NVIDIA's driver (until
2026-11-23 the handles of the first four could not be created, and cuDNN 9.10's first operation
failed: one of cuBLAS 12.9's images has ten grouped GEMM kernels for sm_80 that the layer does
not translate). A program of the CUDA 12 runtime whose own fat binary has a refused kernel
starts without the kernels of that fat binary under eager loading; under lazy loading it is
given the others.

What eager loading costs through the layer: every image of machine code that a runtime
registered is translated, and every kernel of retargeted PTX that it asks for is compiled by the
driver, at its start, and every refused kernel is named then on standard error. With the layer's
cache off (`VF_PROXY_CACHE=0`), cuBLAS 12.9 started in 30 s (2 s without the layer); the start of
PyTorch 2.14 (3,358 images of PyTorch, cuDNN, cuBLAS and the others adapted, 1,411 kernels
named) had not ended after ten minutes, nor that of cuDNN 9.28 (its CUDA 13 build) after two, by
which it had compiled 1,441 kernels one by one (2026-11-23). A later start takes what it can from
the layer's cache and the driver's (`CUDA_CACHE_PATH`); that was not measured.

**PyTorch 2.14.0.** Of the 23,693 functions for sm_75 in the 475 units of machine code of its
library, 23,664 are translated (99.9 %) and 29 are refused: 10 for a branch through a uniform
register, 6 for the instruction `I2IP`, 4 for a call whose return address is not given the
usual way, 4 for a conversion of a uniform register, 2 for `IMAD.HI.X`, 2 for a jump table, 1
for want of registers (`vf translate` of 0.1.3 on each unit, 2026-11-23). None of them is
reached by the tests below.

Of 925 operations compared with the CPU (`tests/gpu/torch.sh`): **924 give the same result**,
1 is skipped (the CPU does not do it), none is refused, none is wrong. Of 32 small whole models
(`tests/gpu/torch-models.sh`): 32 pass.

**PyTorch 2.14.1**, which is what `pip install torch` brought from PyTorch's index for CUDA 13
on 2026-11-21, installed in an environment made by `vf venv`: the same 925 operations and 32
models, with the same outcome (924 and 1; 32). Its machine code was not taken out and counted
as that of 2.14.0 was: the figures above and the timings below are those of 2.14.0.

**FlashInfer** (SGLang's default attention). Of its eighteen prefill kernels
(`BatchPrefillWithPagedKVCache` and `BatchPrefillWithRaggedKVCache`, nine each), thirteen are
translated and five are refused: three that name 255 registers, two for an instruction that
the translator did not have (`LEA.HI` with a uniform register). These are the counts of
2026-11-20; the rule for `LEA.HI` with a uniform register came on 2026-11-21 (in 0.1.2), and
these kernels were not counted again.

**Other libraries.** The machine code of 42 libraries of CUDA 13.5, cuDNN, NCCL and others was
translated and checked instruction by instruction, not run (`docs/validation.md`, "The corpora"):
of 3,108 units, 3,105 are translated in whole or in part; three are not read at all (a section
table that points outside the file; two sections of code of the same name). In these libraries
more functions are refused than in PyTorch (10,665 of 70,975, `vf translate` of 0.1.3,
2026-11-23); cuFFT's kernels for double
precision are among them (relocations that the translator cannot carry with their instructions).
The CUDA 13 libraries are not meant to be translated: the layer puts the CUDA 12.9 builds, which
have code for Volta, behind their names (`INSTALL.md`).

**Cubins for sm_80 and later.** VF can use the PTX that NVIDIA's assembler retains in
`.nv_debug_ptx_txt` with line information. It checks the architecture and the kernel names
before compiling that PTX for Volta. This covers standalone cubins from Triton and cubins in
fat binaries. Calls that the executable leaves to the driver (`__assertfail`, `vprintf`,
`malloc`, `free`) are accepted when the retained source declares them `.extern .func`. For sm_75, VF also tries retained PTX when its default machine-code translation
refuses every kernel in the image. The line information also tells which additions written
without a rounding modifier ptxas fused with a multiplication: one that it did not fuse
(`--fmad=false`, a debug build) is given the `.rn` form, so that the driver's compiler does not
fuse it either, and a source of which that cannot be told is not used.

Since 0.1.2, VF also translates a subset of stripped `sm_80` executables. VF checks each function's
instruction forms, branches and memory addressing. Global loads and stores, including 128-bit
accesses, require the CUDA system descriptor on every path. VF lowers the supported 16-byte
global-to-shared copy to a guarded Volta load and store and waits for the completed group.
SGLang uses this path for its top-eight expert sum and its wheel’s 32-bit expert counting and
alignment kernels. The router uses shared counter increments and GPU-scope 32-bit global atomic adds.
FlashInfer uses the same translator for 2D RMSNorm, fused residual RMSNorm and 3D query/key
normalization. For compiler-generated warp helpers, VF checks the call target, return register
and descriptor writes. Unsupported functions are refused individually. Other stripped targets above
`sm_75`, including `sm_86`, still require compatible PTX or a second supported image. Of 31,207
functions in 1,193 sm_80 cubins of PyTorch, cuBLAS, cuSOLVER and cuFFT, 900 (2.9 %) were
translated (2026-11-22). **This translation is not checked by the independent validator**, which
reads sm_75 code only: what checks it, and what does not, is in `docs/validation.md`, section 8.

**PTX.** A kernel that uses an instruction with no equivalent on Volta is refused: clusters of
blocks, Hopper's barriers, tensor maps (`docs/emulations.md`, "Other instructions").
For an indirect call within a single PTX module, an explicit `.calltargets` list identifies the
possible callees. With only a `.callprototype`, VF follows each device function that
could match the declared input and return lists. It distinguishes ordinary 32-bit and
64-bit scalars. Unfamiliar declarations, including array parameters, leave all targets
possible. A kernel is refused if one of those paths reaches a trapped function.
When linking separate device objects, an indirect call specified only by a prototype
can still be refused if any linked object contains a trapped device function.
The image that the driver's linker makes (`cuLinkComplete`) is known by its bytes, as it is or
inside a fat binary, for the life of the process only: saved to a file and loaded in another
process, its kernels that trap are given to the program, and launched, end its context.
The cubin that the layer's nvJitLink (`libnvJitLink.so.13`) makes from inputs of which some
kernels cannot run (and others can: when none can, the link is refused) names them in a note
(`.note.vf.refused`). The layer reads it wherever the cubin is loaded, in any process: as a cubin,
as an input of the driver's linker, or stored uncompressed in a fat binary. Stored compressed in a
fat binary, it is not read (that would take decompressing the code of every library at every
load), except in a process whose links through the driver had kernels to refuse. A cubin loaded without the layer keeps its kernels that trap: the warning that nvJitLink
writes when it links them is then all there is.

**A variable asked for by name can be refused too.** In a library (`cuLibraryLoadData`, which
the CUDA runtime uses), a kernel compiled on its own (`VF_LAZY`, the default) that reads a
lookup table of its module (a variable with an initializer and no `.visible` or `.weak` name,
which no function of the module can change, as a `static const __device__` array or the grids
of the i-quants in llama.cpp and Strata) is given a copy of the table. NVIDIA's driver gives
such a table by its name in the PTX (`table`, or the name that nvcc mangles in a namespace), and
a kernel then reads what the program writes there. Through the layer, once a kernel holds a
copy, a lookup of the table by name returns `CUDA_ERROR_NOT_SUPPORTED` (801) instead of an
address, and a line on standard error says why, since a write through the address would not
reach the copy: `cuLibraryGetGlobal`, `cuModuleGetGlobal` on a module of the library, and with
them the CUDA runtime's `cudaMemcpyToSymbol`, `cudaMemcpyFromSymbol` and
`cudaGetSymbolAddress` (`cudaErrorNotSupported`, measured with CUDA 13.5). The answer depends on
the order: a table that the program asks for before any kernel that reads it is given, and its
kernels then come from the whole module, where the program's writes are. A name that the module
does not have is `CUDA_ERROR_NOT_FOUND` (500), as without the layer. `VF_LAZY=0` compiles every
module whole, with one copy of each variable, and gives every table. Neither Strata on two
V100s (issue #3) nor any case of `tests/gpu/run.sh` but those that test this asked for such a
table (2026-11-23).

## Kernels launched from a second translation

A kernel compiled for blocks of 1,024 threads may have 64 registers at most on a V100, and the
usual translation takes some kernels over that. When the driver refuses a launch for it, the
layer translates the module again within each kernel's own count and launches the kernel from
there (`VF_FIT=0` turns this off). In PyTorch's tests, `avg_pool2d` and `interpolate` in
float32 and float64, `unfold` in float32 and three of the whole models need it.

- The independent validator finds nothing in PyTorch's 475 units of machine code translated
  that way, as in their usual translation (`docs/validation.md`, "The corpora": until
  2026-11-20 it objected to 55 of them; five kinds of objection were rules of the validator
  that did not fit a translation within a kernel's own registers, and one was a fault of the
  translation, corrected). The other corpus of machine code, 42 libraries, is at 0 problems
  translated that way too.
- That second translation is slower code; its speed was not measured against a native build.
- It is not used for a kernel that can reach data of its module, nor for a kernel of a module
  from which the program asked for a variable or a texture or surface reference by name
  (`cuModuleGetGlobal`, `cuModuleGetGlobal_v2`, `cuModuleGetTexRef`, `cuModuleGetSurfRef`,
  `cuLibraryGetGlobal`, `cuLibraryGetManaged`, and what `cuGetProcAddress` gives for
  them): such a launch is refused to the program, as the driver refused it. A library and the
  module that the driver gives for it (`cuLibraryGetModule`) are one image: a variable asked
  for through either excludes the kernels taken through the other.
- It covers the four launch entries (`cuLaunchKernel`, `cuLaunchKernelEx` and their forms for
  the per-thread default stream), called or taken from `cuGetProcAddress`, and kernels taken
  with `cuModuleGetFunction`, `cuLibraryGetKernel`, `cuLibraryEnumerateKernels`, or
  `cuModuleGetFunction` on the module of a library.
- What the program sets on a kernel (`cuFuncSetAttribute`, `cuKernelSetAttribute`, the cache
  configuration) is set on its second translation too. A library's kernel and the function of
  the same name in the module that the driver gives for the library are one kernel to the
  driver (measured on a V100: a setting made with `cuKernelSetAttribute` holds for the
  function, unless the function has one of its own for that attribute, whatever the order of
  the calls): the function's second translation is given the library kernel's settings for
  the device of its context, then its own. The other way round is not done: a setting made
  with `cuFuncSetAttribute` on such a function is not carried to the second translation of
  the library's kernel, and a launch through the library's kernel that needs it (more dynamic
  shared memory than the default allows) is refused there as the first was.
- A launch whose second translation was being made when the program asked for data of the
  module by name gets the driver's refusal, like the launches after it: none starts from a
  second translation once the program has been given such data.
- It does not cover a launch recorded in a CUDA graph, a kernel launched through a function
  handle made from a library's kernel (`cuKernelGetFunction`), nor the old launch functions
  (`cuLaunch`, `cuLaunchGrid`).

## Speed

`docs/performance.md` has the method and the detail. In short:

**PyTorch.** 89 operations timed against the wheel of the same PyTorch that has code for sm_70
(2026-11-20 and 21, five runs): from 0.79 to 1.65 times its time, median 1.00, geometric mean
1.02 to 1.03; 69 to 72 of the 89 within 3 %. The slowest: attention in float16 (1.65: Turing's tensor-core
products are emulated), `std` along a dimension (1.3), `argmax` and `max` along a dimension,
`cumsum` (1.10 or 1.24, from one run to the next), a 3x3 convolution and a small CNN (1.2).
The convolutions are slower by the algorithm that the newer cuDNN picks, not by the layer.

**llama.cpp.** With a 0.5B model (2026-11-20): generation one token at a time at 410 to 470
tokens a second, 97.5 to 102 % of the native build's speed; a prompt of 2,048 tokens at 9,700
to 11,900 tokens a second, 40 to 53 % of the native build's speed (for a batch the official
build takes kernels with integer tensor-core products, which are emulated, where the native
build calls cuBLAS). Flash attention, 196 cases of llama.cpp's own backend test (2026-11-18): 1.60
times the native build's time over all of them; most of the worst cases compare two different
kernels, since llama.cpp picks another kernel on a Volta than on the Turing it is told.

**llama.cpp with NVFP4 weights** (Qwen3.8-27B, 2026-11-20): generation at 98.5 % of the native
build's speed; prompts at 299 tokens a second, 1.33 times faster than the native build as it
is (224) and 1.95 times slower than the native build with its quantized kernels forced (582).

**vLLM, SGLang, ZML**: no native build exists to compare with. vLLM 83 to 99 tokens a second
and SGLang 45 to 59 (Qwen2.5-0.5B); ZML 27 to 29 (Qwen3.5-9B). Runs of 2026-11-20 and 21.

**Triton's matrix products** (`tl.dot`). Told 7.5, which is what Triton is told by default in a
program with PyTorch, Triton 3.8 uses no tensor-core instruction: it multiplies with float32
`fma` (its own choice below 8.0; it says so with `MLIR_ENABLE_DIAGNOSTICS=remarks`). Its results
are right and slow: a float16 GEMM at 0.23 to 3.2 TFLOPS, vLLM's mixture-of-experts kernel at
0.1 to 0.15, its attention at 2.6 (prefill) and 0.54 (decode); the same kernels told 8.0 (with
`VF_CAPABILITY_ALL=1`, and its hazards: [The capability told with
`VF_CAPABILITY`](#the-capability-told-with-vf_capability)) at 26 to 33, 4.4 to 12.9, 13.6 and
1.8 (2026-11-23, one session). The layer's translation adds 11 instructions to the 5,952 of
Triton's GEMM kernel: the time is in that code (`docs/performance.md`, "Triton's matrix products").

**Blackwell's block-scaled products** (NVFP4, MXFP4): a matrix-vector product written with them
runs at 2.7 times the time of a kernel written by hand for Volta (2026-11-18).

**The first run** of a program translates its machine code, rewrites its PTX and lets the
driver compile it: seconds to a minute for the programs above, kept in a cache for the next
runs (`INSTALL.md`). The translator's time grows faster than the size of a function: one unit
of machine code of vLLM's compiled extension for mixtures of experts (5 MB, 224,000
instructions in 36 functions; not among what the tests here make the layer translate) takes
it more than seven minutes (2026-11-20).

## Emulated instructions

A V100 lacks these; the layer computes them otherwise (`docs/emulations.md`). A kernel that
runs emulated instructions is announced on standard error.

| Instructions | How | Result | Speed |
|---|---|---|---|
| Turing's tensor-core products on FP16 | Volta's products, operands gathered from other threads | the host model's and an RTX 5090's where the sums are exact; on arbitrary values the last bits differ from an RTX 5090's (47.6 % of random results identical, 41.2 % one unit of the last place away), as a program built for Volta differs | slower: attention in float16 at 1.64 times native |
| Products on bf16, tf32, f64 | each thread computes with `fma`: no tensor core of a V100 takes them | sums added in another order than the hardware's: for bf16 about one random result in five differs from an RTX 5090's, by 1 or 2 units of the last place | slow |
| Integer tensor-core products | `dp4a` on bytes gathered from other threads | exact | 1.95 times a kernel written with `dp4a` (above) |
| FP8 products | exact conversion to halves, then Volta's products | as FP16 products | |
| Block-scaled products (NVFP4, MXFP4, MXF8F6F4) | each block through Volta's products, the scales applied after | NVFP4: of 406,912 results compared with an RTX 5090's, 405,861 identical, 1,051 one unit of the last place away, none further. The other forms agree less (`docs/emulations.md`, "Block-scaled products") | 2.7 times a kernel written by hand |
| Conversions of bf16, tf32, FP8, E2M1, UE8M0 | bit manipulation | every value of the small formats gives what an RTX 5090 gives, byte for byte | |
| bf16 arithmetic and atomic additions | `fma`; a compare-and-swap loop | | |
| `cp.async`, 256-bit loads | plain loads | | |

The layer takes it that a program keeps the rows that `ldmatrix` reads at multiples of 16
bytes, as the PTX manual requires. A program that does not stops with a misaligned-address
fault (error 716); `VF_ALIGNED_ROWS=0` makes the layer test the addresses, with slower
emulated kernels.

## The capability told with `VF_CAPABILITY`

`VF_CAPABILITY=8.0` (or later) is for programs whose kernels are PTX, which the layer then takes
for that capability: llama.cpp, TensorFold, ZML's own library, and the engines whose guides set
it. A caller whose code the layer can only give from its sm_75 machine code (PyTorch, cuDNN,
cuFFT, cuSOLVER, and the CUDA runtime that such a library uses) is still told 7.5, with one line
at the default log level: told 8.0, it would choose kernels that its sm_75 code has as stubs,
which write nothing, and told 9.0, some that stop the GPU (`docs/architecture.md`, section 4).
What that does not cover:

- `VF_CAPABILITY_ALL=1` tells the value asked for all the same to a CUDA runtime and to the code
  that uses it (PyTorch's library), the program; a library that asks for itself without that
  runtime (cuDNN, cuFFT, cuSOLVER) is still told what its own code allows (told 8.0, cuDNN's convolutions gave zeros, cuFFT stopped the GPU
  and cuSOLVER's Cholesky factorization failed). A kernel that the program then chooses for that
  capability, and whose sm_75 code is a stub (it writes no memory, where its code for the
  capability told does: PyTorch's attention kernels for sm_80), is refused, and the operation
  fails. It is only for a program that does not launch those kernels, as SGLang with the
  settings of its guide.
- NVML, answered by the audit library, reports the value asked for to every caller: told 8.0, a
  program that reads NVML (vLLM does) and a CUDA runtime that is told 7.5 give it two answers.
- Triton compiles for the capability that PyTorch reports (`torch.cuda.get_device_capability`)
  whenever PyTorch is installed: Triton 3.8 imports PyTorch while it looks for a GPU, also in a
  program that does not import it. With `VF_CAPABILITY=8.0` alone, Triton is then told 7.5, as
  PyTorch's CUDA runtime is, and its `tl.dot` runs without tensor cores ([Speed](#speed)).
  `VF_CAPABILITY_ALL=1` gives it 8.0 and the tensor cores, and gives PyTorch 8.0 too, with the
  refusals of the first item. Told 8.0, Triton also computes a float32 `tl.dot` of its
  default precision in tf32 (10 bits of mantissa), as on an Ampere GPU, where told 7.5 it
  computes it in float32; a bf16 `tl.dot` told 8.0 runs on the layer's scalar bf16 products,
  unless a `VF_BF16_TC_*` switch is set (`docs/emulations.md`, "bf16"); the `tf32x3` precision
  is refused. With `VF_CAPABILITY=8.0 VF_CAPABILITY_ALL=1`, vLLM's engine (greedy and sampling)
  and HTTP server gave the CPU's text with the small model of `tests/gpu/vllm.sh` (its stages b
  to d, 2026-11-23); `transformers` with its default attention, which is PyTorch's, does not.
  A program that does not itself import PyTorch can keep Triton from importing it with
  `TRITON_DEFAULT_BACKEND=nvidia`: Triton then asks the driver itself, and with
  `VF_CAPABILITY=8.0` it is told 8.0 while the libraries whose code runs from sm_75 machine code
  are still told 7.5.
- `TRITON_OVERRIDE_ARCH=sm80` does not make Triton 3.8 use tensor cores on a GPU it is told is
  7.5: Triton applies it to its own passes, but writes the PTX for, and calls ptxas with, the
  capability it is told. ptxas refuses the result (`Feature '.m16n8k16' requires .target sm_80
  or higher`), and a kernel with more than one stage (Triton's default is three) stops the
  process inside Triton (`LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.cp.async.wait.group`).
- What a caller is told is decided when it first asks. A library loaded after that (an extension
  opened once CUDA is in use) is not looked at, nor a library that opens the CUDA runtime with
  `dlopen` instead of naming it among the libraries it needs.
- A value that is not `<major>.<minor>` (`8`, `80`, `sm_80`), or that is below 7.0, is not used,
  and the layer says so once.

## CUDA graphs

A Tesla V100 can capture and replay translated kernels. The official SGLang AWQ server used CUDA graphs for both prefill and decode. A graph can report error 701 when it is instantiated or replayed if the translated kernel needs too many registers for the recorded block size. For translated machine code, `VF_BORROW=9 vf …` can preserve a source block size that would otherwise be lost. Shared-memory limits still apply. [SGLang measurements](docs/guides/benchmarks/sglang-default-operators.json) · [GPU validation](docs/validation.md#cuda-graph-replay)

## The bf16 cuBLAS variants

What `VF_CUBLAS_BF16_TC=1` and `VF_CUBLAS_BF16_SKINNY=1` change is in `INSTALL.md` and
`docs/emulations.md` (section 11). Both are off by default. Their limits:

- **The tensor-core path is for inference, not for training.** Its results are not cuBLAS 12.9's
  bits: they differ in the last bits (`docs/emulations.md`, section 11). It makes bf16 training
  slower: the products of the backward pass fall into its fallback (below, "Speed that depends on
  the data"), and a LoRA training step ran at 0.56 to 0.81 times its speed without the switch,
  full fine-tuning at 0.37 to 0.66 times. Eager PyTorch inference programs ran 1.3 to 1.7 times as
  fast with it (prompts, generation at batch 8 and 32, an embedding encoder; Whisper 1.24 to 1.30,
  generation at batch 1 unchanged), and up to 1.9 times under `torch.compile`
  (`docs/performance.md`, "bf16 products through cuBLAS: the opt-in variants on real programs":
  one model of each kind, on one V100). Not in this release: a gate on k (the path takes every
  product of its shapes up to k = 16384), and a safe mode that keeps every call at 12.9's bits
  whatever `VF_CUBLAS_BF16_TC` says (`VF_SAFE`, which this release does not read).
- **Graph captures.** A tensor-core call captured in a graph is computed by cuBLAS 12.9: its
  workspace would be an allocation node of the program's graph, and with one, measured on driver
  580.178.04 (`tests/gpu/forward/bf16-graphs.c` with the test build's control
  `VF_BF16_CONTROL=capture-tc`, which serves captured calls), a second instantiation of the graph is
  refused, it cannot be embedded as a child graph or instantiated for device launch,
  `cudaGraphExecUpdate` from a recapture is refused, and a graph captured into a conditional node's
  body cannot be instantiated at all, while the same graph without the node works; the layer cannot
  tell any of these apart when it is called. A program that replays its steps from graphs gains from
  the tensor-core path only in what runs outside them: vLLM replays its decode steps and its prefill
  chunks up to the largest captured size (512 and 2048 tokens for the two models measured), and with
  `VF_CUBLAS_BF16_TC=1` its prefill was 6 % faster on one model and unchanged on the other. The skinny
  path serves captured calls (one kernel node, nothing allocated). A replayed tensor-core call has
  12.9's bits, which differ in their last bits from the same call made outside a capture: a program
  that compares a graph's replay with an eager run of the same step sees it.
- **Calls not served**: one column (n = 1), batches of more than one matrix, the other entry points,
  explicit algorithms of `cublasGemmEx` and `cublasGemmStridedBatchedEx`, a `cublasLtMatmul`
  algorithm on the skinny path other than the one 12.9's heuristic gives there (id 16 with every
  configuration attribute as `cublasLtMatmulAlgoInit` leaves it, k in one split),
  pedantic modes, k above 16384 on the tensor cores, cuBLASLt descriptors with any attribute outside
  the supported set, a `cublasLtMatmul` without a C descriptor, with a workspace size and no
  workspace, or with a workspace that is not 256-byte aligned (12.9 refuses that), a bias epilogue
  whose bias overlaps D, and every call of a context that is not a V100's. A legacy workspace set at
  an address that is not 256-byte aligned counts as smaller than 4 MiB (with 4 MiB 16 bytes past an
  aligned address, 12.9 ran its gemmSN kernel at TN 448 x 8 x 256). The tensor-core path does not follow
  a `cublasLtMatmul` algorithm: it computes the product its own way, whatever the algorithm says.
- **Speed that depends on the data.** A row of op(A) or a column of op(B) that does not convert
  exactly to fp16 is computed by the fallback, which is slower: with every column failing at
  n = 480, a tensor-core call took 32 to 71 % longer than cuBLAS 12.9's on six shapes of linear
  layers, TN m x k = 1536 x 1536, 4608 x 1536, 1536 x 4608, 7344 x 1536, 9728 x 896 and 896 x 4864
  (`tests/gpu/forward/bf16-cost.c`), against 1.7 to 2.8 times as fast when nothing fails.
- **What a program sees of the layer's own work**: cuBLAS's logging (`CUBLASLT_LOG_LEVEL`) shows the
  layer's internal fp16 products; a framework's allocator does not see the layer's workspace, which
  the pool of the context keeps up to 64 MiB between eager calls.
- **cuBLAS's workspace contract is unchanged.** A forwarded call uses the handle's workspace as 12.9
  does: threads that share a handle and its workspace on different streams race in cuBLAS itself, as
  they do without the layer (`tests/gpu/forward/bf16-threads.c` gives each calling thread a context
  of its own; calls the layer serves use workspace of their own).
- **The skinny path's rule** was measured with CUDA 12.9.2's cuBLAS (build 12.9.2.10) on one V100 and
  one driver. The runtime checks that cuBLAS reports 12.9.2 (the build number cannot be read at run
  time); with another cuBLAS the skinny path serves nothing.
- **Contexts and processes.** A CUDA context gets the runtime's module at the first call either path
  serves in it, and a memory pool and a cuBLASLt handle at the first call the tensor-core path serves
  there; none is destroyed: a program that makes and destroys many contexts keeps one of each per
  context where they were made (the pool up to 64 MiB). That first call can come late, when device
  memory is nearly full: a module that fails to load then is not loaded again, and that context forwards
  every call for the rest of the process, with one line at level 1 (a pool or cuBLASLt handle that
  cannot be made: its tensor-core calls). A handle used from another context than its own is forwarded (by
  the code; not tested on a GPU). Two V100s in one process, and a V100 beside another GPU, were not
  run on real hardware: the host test, with a stand-in driver whose two devices have different compute
  capabilities, shows the decision taken per context's device at each call, every call of the other
  device's context forwarded. After a `fork`, a child that calls through the variants gets status 13
  and then `cudaErrorInitializationError`, as a child that calls cuBLAS 12.9 directly does, and the
  parent's next call is served with 12.9's bits (`tests/gpu/forward/bf16-default.c`, mode `fork`, run
  by hand).

## Several GPUs and NCCL

Any use of several GPUs from PyTorch goes through NCCL: `torch.distributed` (a process for each GPU), `torch.nn.DataParallel` (one process with several GPUs) and SGLang's tensor parallelism (below). The NCCL of the CUDA 13 wheel cannot run its collectives on a V100: the layer refuses its kernels, whose indirect calls can reach multicast instructions with no Volta translation. NCCL's CUDA 12 build has code for Volta: fetch it with `vf fetch --only libnccl` and start the program through `vf`, which gives it that build in place of the wheel's. Without it, on two V100s, `init_process_group` succeeds and the first collective fails after a minute and a half (`Cuda failure 'a PTX JIT compilation failed'`), and `DataParallel` stops when it copies the model to the other GPUs (`NCCL Error 1: unhandled cuda error`); with its first refusal of an NCCL kernel the layer says that the CUDA 12 build is missing.

SGLang starts one process per GPU for tensor parallelism. A user tested two and four V100s with NCCL's CUDA 12 build mapped through VF. With the CUDA 13 build, the first all-reduce failed. VF reported that NCCL's indirect calls could reach multicast instructions with no Volta translation. Four ranks also needed the uniform-address translation merged in PR #1. [Issue #2 and reproduction](https://github.com/nbald/volta-forever/issues/2) · [Address translation change](https://github.com/nbald/volta-forever/pull/1)

We separately tested this mapping with two Tesla V100-SXM2-32GBs. One process requested NCCL by soname and the other used the CUDA 13 library's absolute path. In a control with NCCL 2.30.7 for CUDA 13, VF refused `AllReduce_Sum_f16_RING_LL` on both ranks and the run timed out during initialization. With the map, both ranks loaded NCCL 2.32.6 for CUDA 12. Each rank received the exact expected integer sums for 1, 257 and 65,536 elements. [Commands for the SGLang environment](docs/guides/sglang.md#6-additional-notes)

Four PyTorch ranks with the CUDA 12.9 build of NCCL also passed SUM and MAX reductions on four V100s. All 1,052,704 output values matched a CPU oracle, and NCCL reported peer access in all twelve directions. The full SGLang 35B run completed on two GPUs. The four-GPU model startup stopped before generation. [Collective and model checks](docs/guides/benchmarks/sglang-tensor-parallel.json)

## Peer copies over NVLink

One process enabled direct peer access between two V100-SXM2-32GBs connected by two NVLinks (`NV2`). It copied a 64 MiB buffer and verified every byte in each direction. Each rate below is the median of three sets of 32 copies after warm-up, measured one direction at a time on driver 580.178.04.

| Bulk peer copy | Native driver (GB/s) | With VF (GB/s) |
|---|---:|---:|
| GPU 0 → GPU 2 | 48.44 | 48.45 |
| GPU 2 → GPU 0 | 48.44 | 48.45 |

The timer covers the copies through stream synchronization. The driver did not expose per-link traffic counters on these cards. These rates describe application payload on this GPU pair. [Test source](tests/gpu/peer-copy.c)

Build with the CUDA headers and run on a pair listed as `NV2` by `nvidia-smi topo -m`. Set `CUDA_HOME` to your toolkit directory.

```sh
gcc -std=gnu11 -O2 -I"$CUDA_HOME/include" tests/gpu/peer-copy.c -lcuda -o /tmp/vf-peer-copy
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2 /tmp/vf-peer-copy
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2 vf /tmp/vf-peer-copy
```

## What was not tried

- A suite of `tests/` with several GPUs in one process. The review of 2026-11-22 ran PyTorch on
  four V100s in one process, and the driver API from four threads on four devices, with no wrong
  result; no script of the repository does.
- A machine with a V100 and a later GPU. Modules loaded into a context on the later card are
  given to the driver as the program gave them; this is tested with a stand-in for the driver
  only. A library (`cuLibraryLoadData`) is loaded for every card: its fat binary keeps the entries
  that are replaced for the V100 beside their replacements, and the later card takes its own. A
  kernel refused for the V100 is given in the module that the driver gives for the library in a
  context on the later card, and refused when it is taken from the library (`cuLibraryGetKernel`,
  which is for every card). A library whose PTX is compiled kernel by kernel (deferred) gives the
  later card the V100's PTX.
- MPS (the Multi-Process Service)
- Driver 580.196.07 and later drivers of the 580 branch (everything was run with 580.178.04)
- Windows; ARM
- CUDA 14 and later
- Nsight Compute and Nsight Systems (CUPTI's sampling of the program counter was used through
  the layer)
- Coverage across all model architectures. The engine guides give the checkpoints and workloads tested.

## Numerical accuracy

- **Machine-code translation** does not change arithmetic, Turing's tensor-core products apart
  (the table above): a translated kernel runs the program's own operations in the program's
  order, on other registers. The tests compare its results with the native sm_70 build of the
  same source where there is one (60 kernels, the same results), and with the CPU otherwise.
- **PTX**: integer products are exact. FP16 products are Volta's own, which add four terms
  at a time in float32: they give the host model's results where the sums are exact, and on
  arbitrary values differ from a later GPU's in the last bits, as a program built for Volta
  does (`docs/emulations.md`, "Products of halves and floats of later shapes"). bf16 and tf32
  products are computed by each thread with successive `fma` in float32, where the hardware
  keeps a wider sum and cuts it at the end: about 20 % of random bf16 results differ from an
  RTX 5090's by 1 or 2 units of the last place.
- **Conversions of the 4-bit and 8-bit formats**: identical to an RTX 5090's on every value of
  the formats.
- **Block-scaled products**: see the table above.
