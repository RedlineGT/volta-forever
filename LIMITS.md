# volta-forever: known limits

What does not work through the layer, what is slower, and what was not tried. Everything was
measured on one Tesla V100 with driver 580.178.04 and no switch set; a figure carries its date
where it is not from the build of this release. `docs/` says how each figure was obtained:
`docs/validation.md` for what was compared with what, `docs/performance.md` for the timings,
`docs/emulations.md` for what is computed in place of the instructions that a V100 lacks.

## The limits of 0.1.5 to know first

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
  (2026-11-23, with the translator and the rules of 0.1.4). A translation that stored
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
- **A refused kernel that a program obtains without a lookup is refused where it is used, and
  some uses still trap.** The layer refuses a kernel when the program looks it up; a handle that
  the program holds all the same (cuFFT enumerates the functions of its modules through a private
  table of the driver) is refused at a launch, in a kernel node of a graph made or changed with
  it, and in a question about it or a setting of it: error 218, and a line names the kernel. What
  still traps: a kernel launched by another kernel (dynamic parallelism), a function reached by an
  indirect call or through a call into another linker input, and a launch through a private
  interface of the driver ([Kernels refused](#kernels-refused)).
- **A kernel whose code is a trap and nothing else is given, and its launch refused.** A kernel
  built for some architectures only is emitted for the others with a body that traps; launched,
  it would end the context (Xid 43). Such a kernel of PTX that the layer retargets or of sm_75
  machine code that it translates is given at the lookup, as NVIDIA's driver gives it, but every
  launch of it and a kernel node of a graph with it return 218, and a line names the kernel and
  its image: a program that chooses such a kernel for the V100 stops at that error, without a GPU
  fault. Strata's CUDA 13 engine 0.1.66 does, with its default settings, at its first prompt of a
  few thousand tokens (4,021 here); its guide says which setting avoids it ([its
  guide](docs/guides/strata.md)). Not recognised: such a kernel in an input of the driver's
  linker (`cuLinkAddData`) or of nvJitLink, which traps as without the layer, and code that an
  image has for sm_70, which is loaded as it is. A kernel whose body is empty is not refused:
  launched, it does nothing and returns no error, as without the layer; `VF_LOG_LEVEL=2` counts
  them in each image adapted ([Kernels refused](#kernels-refused)).
- **The bf16 cuBLAS variants are opt-in.** `VF_CUBLAS_BF16_SKINNY=1` and `VF_CUBLAS_BF16_TC=1`
  are off by default. The tensor-core path's results are not cuBLAS 12.9's bits, and it takes TN
  products with k from 768 to 16384 only, but for the classes of shapes that its rules of speed
  leave to 12.9: a model whose layers have fewer than 768 in-features gets it only on its wider
  layers. A bf16 fine-tuning step ran at 1.12 to 1.24 times its speed without it, its backward
  products going to 12.9 (0.1.5 served them too, and made such a step slower). `VF_SAFE=1` turns
  it off ([The bf16 cuBLAS variants](#the-bf16-cublas-variants)).
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

A program can hold a handle to a refused kernel without having asked for it: cuFFT enumerates
the functions of each module it loads through an entry of a private table of the driver
(`cuGetExportTable`), not by their names, and an enumeration made before the driver's compiler
refused something of a deferred library still holds what it gave. The function is in the module
that the driver loaded, made to trap. Such a handle is refused where it is used: a launch
(`cuLaunchKernel`, `cuLaunchKernelEx` and their forms for the per-thread default stream,
`cuLaunchCooperativeKernel` and `cuLaunchCooperativeKernelMultiDevice`, the old `cuLaunch`,
`cuLaunchGrid`, `cuLaunchGridAsync`), a kernel node of a graph made or changed with it, a
question about it or a setting of it (`cuFuncGetAttribute`, `cuFuncGetParamInfo`, the occupancy
functions and the like) and `cuKernelGetFunction` return 218, the driver is not called, and a
line names the kernel and the reason once. What still traps: a kernel launched by another kernel
(dynamic parallelism), a function reached from a kernel by an indirect call or through a call
into another linker input (above), a launch through a private interface of the driver (cuFFT
launches through `cuLaunchKernel`; no other is known).

**A kernel without code is given, and its launch refused.** A kernel built for some
architectures only is emitted for the others too, with a body that is a trap and nothing else.
Launched, it ends the context on any GPU: with NVIDIA's driver and the sm_70 code of one, the
launch returned 0, the next synchronisation 719 (`CUDA_ERROR_LAUNCH_FAILED`), and the GPU wrote an
Xid 43 line (2026-11-24). Through the layer, such a kernel of PTX that it retargets or of sm_75
machine code that it translates is given when the program asks for it and answered about, as
NVIDIA's driver does, but a launch of any kind and a kernel node of a graph return 218 without
calling the driver, and a line names the kernel and its image; the context goes on. Code that the
image already has for sm_70 is loaded as it is and not read: a stub there traps as without the
layer. Nor is what was found carried through a link: a kernel without code in an input of
`cuLinkAddData` or of nvJitLink is launched from the image that the link makes, and traps, as
without the layer (at `VF_LOG_LEVEL=2` the line of an input of `cuLinkAddData` says that its
launch is refused, which it is not). A kernel with an empty body is not refused: launched, it does nothing, as
without the layer, and nothing says so but `VF_LOG_LEVEL=2`, which counts both kinds in each
image adapted. A program that chooses, for the V100 it detects, kernels that its CUDA 13 build
carries only as such stubs now stops with that error and no GPU fault: Strata's CUDA 13 engine
0.1.66 ([its guide](docs/guides/strata.md)).

A kernel can be refused after it was given, though it runs: the layer learns a refusal after it
gave a kernel only when the driver's compiler refuses a function of a deferred library loaded
whole, of which the kernel was given compiled alone (the compiler accepted it then: that code holds
no trap). From then on the kernel is refused at every use, while a kernel node of a graph made with
it before goes on launching it (a graph is not looked at when it is instantiated or launched): the
answers are not the same before and after, but nothing that traps is launched.

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

- **The tensor-core path's results are not cuBLAS 12.9's bits**: they differ in the last bits
  (`docs/emulations.md`, section 11), in inference and in training. What it gains depends on the program's shapes.
  Measured on one model of each kind on one V100 (`docs/performance.md`, "bf16 products through cuBLAS: the opt-in
  variants on real programs"): eager generation at batch 8 and 32 ran 1.40 to 1.43 and 1.49 to 1.52 times as fast
  with it, prompts and scoring 1.33 to 1.74 times, an embedding encoder 1.53 to 1.77 times, Whisper's encoder 1.31
  times and its transcription at batch 16 1.26 to 1.29 times, the generation model under `torch.compile` 1.46 to
  1.92 times, and generation at batch 1 (the steps' one column never served, the prompt's products served) at 1.00
  to 1.04 times its speed without the switch. A bf16 fine-tuning step (LoRA and full fine-tuning of the same 0.5B
  model) ran at 1.12 to 1.24 times its speed without the switch: its forward products are served, those of its
  backward pass go to 12.9 (below, the TN layout). `VF_SAFE=1` keeps every call at 12.9's bits whatever
  `VF_CUBLAS_BF16_TC` says ([Safe mode](#safe-mode-vf_safe1)).
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
  `VF_CUBLAS_BF16_TC=1` its prefill was 6 % faster on one model and unchanged on the other (measured
  before the gate on k). The skinny
  path serves captured calls (one kernel node, nothing allocated). A replayed tensor-core call has
  12.9's bits, which differ in their last bits from the same call made outside a capture: a program
  that compares a graph's replay with an eager run of the same step sees it.
- **One element's bits depend on the batch size, and on capture.** With the tensor-core path on, the
  same row of op(A) times the same column of op(B) has cuBLAS 12.9's bits when the product is narrow
  (n = 1, or within the skinny path's rule: n <= 16, or n < 32 when m k < 2^20) and the tensor-core
  path's when it is wide (m k >= 2^20 or n >= 32, outside the skinny path's rule, with k from 768 to
  16384) and made outside a graph capture; captured, the same call has 12.9's bits again. `VF_SAFE=1`
  removes what the variants add: it gives cuBLAS 12.9's bits for that call, whatever n and whatever the
  capture ([Safe mode](#safe-mode-vf_safe1)). 12.9's own results depend on n at n = 1 and, on some
  shapes, up to n = 16. Measured with
  12.9 alone on 18 TN shapes of linear layers (each column against the same column at n = 64): every
  shape gave other bits at n = 1 (in 77 to 95 % of the elements); TN 384 x 1536 and 256 x 2048, where
  12.9 runs another kernel up to n = 16, gave other bits in about 90 % of the elements at every n from
  2 to 16 and the same bits from n = 17; the 16 others gave the same bits at every n from 2 to 63.
- **The tensor-core path takes k >= 768 only** (`docs/emulations.md`, "The tensor-core path's gate"): a model
  gains on its layers with 768 in-features or more; a shorter product goes to the skinny path (n <= 16, within its
  rule) or to cuBLAS 12.9. On real operands at k = 768 and 896 (the layers of k = 768 and 896 of five models:
  Qwen2.5-0.5B, T5-base, OPT-125m, bge-base-en-v1.5 and whisper-small's encoder, 990 pairs against the exact
  product) every element's error was within 0.040 times the acceptance test's bound in fp32 output, and every
  matrix's mean error towards zero within 0.046 times its bias bound (0.82 units at most). Data whose products all
  have one sign and spread over many binades fare worse below k = 1024: two such families of the acceptance test,
  log-normal magnitudes and an adversarial positive construction, ran at 1.16 and 1.25 times the bias bound at
  k = 768 (0.91 and 1.003 at 896): what a program whose operands are all positive over a wide range should expect
  at k = 768 to 1023, a mean error towards zero above that bound (the bound being 0.8 times an RTX 5090's own on
  the same data), each element still within the error bound. The test judges these two families' bias from
  k = 1024 (documented exceptions, as the two constructions aligned with the path's k-slices are from 2048 and
  4096). `VF_SAFE=1` keeps every call at 12.9's bits ([Safe mode](#safe-mode-vf_safe1)). The skinny path is
  unaffected.
- **The tensor-core path serves the TN layout only** (op(A) = A^T, op(B) = B: a framework's linear layer,
  y = x W^T, as cuBLAS gets it), a rule of speed. Inference keeps the path on the linear layers it gates in,
  and training on its forward products; the products of a backward pass, input gradients (NN) and weight
  gradients (NT, whose k is the batch's tokens), go to cuBLAS 12.9, and so do TT products. In the other
  layouts the path reads strided vectors, 64 to a block of its scale kernel and again for every chunk of
  columns, and a gradient's rows and columns often fail the exact check and go to its generic fallback. Every
  product of four fine-tuning steps (adapters of rank 16 and full fine-tuning, 2,048 and 4,096 tokens, a
  model of hidden size 896), timed in place, each call of a real step waited for: with the rule removed (a
  test build's control), 20 of the 22 NN and NT products the path would serve ran at 0.24 to 0.92 times
  12.9's speed (the other two, of 16 rows, at 1.01 and 1.35), and the steps' products together at 0.61 to
  0.84 times; with the rule, at 1.02 to 1.08 times, the TN products it serves at 1.45 to 3.88 times
  (`docs/performance.md`, section 7).
- **Narrow products of tall matrices go to cuBLAS 12.9**: TN with m >= 8192 is served only with n > 64, and TN
  with m >= 6144 only with k <= 9216 (an output layer over a vocabulary, or a wide MLP's up projection, at a batch
  of up to 64 tokens; and long products of many rows). With so many rows 12.9 fills the GPU. Up to 64 columns, its
  tile's width, it computed such products about as fast as the path or faster from m = 8256 on (1.014 to 1.03
  times as fast at m = 8256 to 8320, as fast or faster from 8384 on, down to 0.65 times its speed on output
  layers): on the layers of the models measured that class ran at a median of 1.07 times 12.9's speed and at 0.66
  at worst, below the class criterion of the rules of speed (`docs/performance.md`, section 7, "The rules of
  speed"). From 65 to 128 columns the path was faster at most points measured (up to 1.9 times), and within 3 % of
  12.9 or slower at some in an earlier sweep (down to 0.95 times, at 112 to 128 columns on output layers and at m
  = 10240): that class passes the criterion (a median of 1.24 and 1.04 at worst on the layers measured; on a sweep
  of that tier at m = 8192 to 248320 and k = 1024 to 4096, a median of 1.40 and 1.05 at worst) and is served. The
  long products of many rows (m from 6144 on with k beyond 9216) have no layer of a real model measured, and a
  class without one goes to 12.9: with operands that convert exactly, 12.9 was within 3 % of the path or faster in
  narrow bands of m and n in which its 64 x 64 tiles fill its waves (48 of the 951 points measured, down to 0.94
  times) and the path faster at the other widths of the same shapes (up to 1.9 times; the median of the 951
  points, 1.30); with 30 to 60 % of the input's columns failing, on synthetic operands, such shapes ran at 0.52 to
  0.89 times 12.9's speed (measured in review). With k up to 9216 the path was at least 1.05 times as fast at
  every point measured.
- **Wide down projections at many columns go to cuBLAS 12.9**: TN with k > 8192 is served only with n <= 256. Such
  products are the down projections of SiLU-gated MLPs, k their intermediate size: their input, the gated
  activation, holds values far below each column's largest, and on real text up to 75 % of a layer's columns fail
  the exact check; the fallback computes them, one fused multiply-add chain over the whole k each, which costs more
  than the path saves from a few hundred columns on. Whether a layer loses depends on its data, which the rule
  cannot see: of the down projections of four models measured on real text (k = 4864 to 12288, n = 32 to 2048),
  the layer that lost (k = 8960, 53 to 70 % of its columns failing) ran at 1.15 to 1.17 times 12.9's speed at 256
  columns and below 1.0 at every n from 384 on (down to 0.62 times; at 320, 0.85 or 1.08 by the choice of
  columns), while the other down projections of k > 8192 ran at 1.05 to 2.15 times 12.9's speed from 320
  columns on; the class goes to 12.9, and with it those gains. No model
  measured has a k between 4864 and 8960; the down projections of k = 4864 kept at least 1.08 times 12.9's speed
  at every n. On four ReLU and GELU models no token vector failed the check.
- **Calls not served**: one column (n = 1), batches of more than one matrix, the other entry points,
  explicit algorithms of `cublasGemmEx` and `cublasGemmStridedBatchedEx`, a `cublasLtMatmul`
  algorithm on the skinny path other than the one 12.9's heuristic gives there (id 16 with every
  configuration attribute as `cublasLtMatmulAlgoInit` leaves it, k in one split),
  pedantic modes, k above 16384 or below 768 on the tensor cores, products in another layout than TN on the
  tensor cores, and TN with m >= 8192 and n <= 64, with m >= 6144 and k > 9216, or with k > 8192 and n > 256
  there, cuBLASLt descriptors with any attribute outside
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
  (`tests/gpu/forward/bf16-cost.c`), against 1.7 to 2.8 times as fast when nothing fails (measured
  before the gate on k, which sends 9728 x 896 to 12.9). From `VF_LOG_LEVEL=2` the report at exit says
  how many tensor-core calls had rows or columns failing the check, how many of them more than a quarter
  of their rows or of their columns, and the share of the calls' elements that the fallback computed;
  `VF_PROFILE` gives the fallback kernel's own time (`vf_bf16_fallback_tn`).
- **A served product can run below cuBLAS 12.9's speed**, whatever the rules of speed above. They serve a class of
  shapes when it has real members and, over every layer and n measured in it, its median speed was 1.10 times
  12.9's or more and its slowest 0.90 or more, and no model measured had its products' summed time above 12.9's at
  any n (`docs/performance.md`, section 7, "The rules of speed", gives each class). That floor holds for the
  products measured, not for every product of a served class: on the real operands of the eight models measured
  every product the rules serve ran at 1.04 times 12.9's speed or more but in one layer, and on the sweeps of the
  served classes at 0.95 or more. What takes a product below it is its data, not its shape: an input with many
  columns failing the exact check, which the fallback computes, one fused multiply-add chain over the whole k for
  each element of a failing column. At long k such a product can run well below the floor, down to about half of
  12.9's speed: on synthetic operands with 30 and 60 % of their input's columns failing (measured in review; 12.9
  and the path alternated in one process, 3 rounds; a failing column made by one element at 2^-40), served shapes
  ran at 0.83 and 0.74 times 12.9's speed (TN 4096 x 256 x 12288, the 9B hybrid's down projection's shape), 0.67
  and 0.55 (TN 5120 x 256 x 13824), 0.77 and 0.70 at 128 columns with 60 % failing, and the 4B hybrid's 2560 x 256
  x 9216 at 0.84 with 60 %; where the path's margin with converting inputs is wide it stayed faster (the 1.5B
  model's 1536 x 256 x 8960, 1.18 with 60 %). The real layers measured at long k had few failing columns (at most
  2 % in the 9B hybrid's down projections, 3 % in the 4B hybrid's; the 1.5B model's layer with 53 to 70 %, of k =
  8960, is in the wide classes and goes to 12.9 from 257 columns), and the report at exit counts the calls with
  more than a quarter of their rows or columns failing (above; in review it counted each of the 338 synthetic
  calls). The one real layer measured below the floor is the documented exception, its own row in the class table
  (its model's summed step stays at 1.488 times 12.9's speed or more at every n measured with it served): a
  product whose input has a quarter or more of its columns failing, near the top of one of 12.9's waves.
  12.9 computes a bf16 product in tiles of 64 x 64, four per SM, 320 at a time on a V100:
  its time is flat over ranges of n and jumps where one more of its tiles lands on the busiest SM (at m =
  4096: at n = 129, 193, and 321, where its second wave begins). The tensor-core path's own time grows
  evenly with n, and the fallback adds one fused multiply-add chain over the whole k for every element of
  a failing column, less efficient per column than 12.9's tiles. With a quarter or more of the columns
  failing, the sum can pass 12.9's time on its flat stretches, most before its second wave. Measured on a
  layer of TN 4096 x n x 4096 whose input had 26 to 38 % of its columns failing, 12.9 and the path
  alternated in one process per n (2 passes): at worst 0.86 times 12.9's speed (n = 320), 0.86 to 0.97 at
  n = 128, 192 and 256 to 320, 1.00 to 1.03 at n = 96, 160 and 640, 1.05 to 1.44 at the other n from 32 to 1024; an attention
  layer of the same model and shape, whose columns all converted, at 1.33 to 2.46 (`docs/performance.md`,
  section 7, gives the table per n with 12.9's waves). No rule of shape tells such a layer from the others
  of its shape, and a choice made from the data would need a synchronization with the host for every call
  (measured: 11 % of the speed of eager decoding at batch 32) and would make an element's bits depend on
  other columns: the path serves it. The report at exit counts the calls with more than a quarter of their
  rows or columns failing (above).
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
  which is for every card); a handle to it that the program holds all the same is launched in a
  context on the later card and refused on the V100. A library whose PTX is compiled kernel by
  kernel (deferred) gives the later card the V100's PTX.
- MPS (the Multi-Process Service)
- Drivers of the 580 branch other than 580.178.04, with which everything was run: 580.173.02, and
  the drivers after 580.196.07. 580.196.07 was reported by users, not run by the project: one report
  has `tests/gpu/run.sh` passing its 165 cases at 0.1.4, another an engine that ran there until a bug
  of its own, on long prompts.
- Windows; ARM
- CUDA 14 and later
- Nsight Compute and Nsight Systems (CUPTI's sampling of the program counter was used through
  the layer)
- Coverage across all model architectures. The engine guides give the checkpoints and workloads tested.

## Safe mode (`VF_SAFE=1`)

What `VF_SAFE=1` turns off, and how it is read, is in `INSTALL.md` ("Environment variables"). It
gives the bits of the conservative reference for what the layer chooses, not results that are the
same everywhere:

- **The reference has dependences of its own.** cuBLAS 12.9 chooses its kernel by the shape of a
  product, so the bits of one column can change with the number of columns (the batch: at n = 1,
  and on some shapes up to n = 16, measured in [The bf16 cuBLAS
  variants](#the-bf16-cublas-variants)); the
  layer's emulated products round otherwise than a later GPU's ([Numerical
  accuracy](#numerical-accuracy)). Safe mode keeps these: a model's outputs can still depend on
  its batch size, and differ from another GPU's.
- **What a program is told is unchanged.** A program told 8.0 (`VF_CAPABILITY`,
  `VF_CAPABILITY_ALL`) chooses its code for 8.0 and computes what that code computes: Triton's
  float32 `tl.dot` in tf32, for one ([The capability told with
  `VF_CAPABILITY`](#the-capability-told-with-vf_capability)).
- **Pairs of products in machine code are not made.** In sm_75 machine code, two products on one A
  computed together (a pair) add the halves of k, (C + k 0..3) + k 4..7, where a product translated
  by itself adds the even k, then the odd k (`docs/pairs.md`, section 9), and their last bits
  differ (every word of the `pair-bits` cases of `tests/gpu/run.sh`, with B loaded plain and
  transposed). By default a kernel gets pairs where it has the registers for them, and none at the
  third level of borrowing (`docs/translator.md`, "Registers"): a kernel that names nearly every
  register, one that requires a block (`.reqntid`) that its translation with pairs cannot have, one
  over its count with `VF_BORROW=9`. Safe mode translates each product by itself at every level,
  which gives these one result (the same bits at levels 0 and 3, measured), whatever `VF_BORROW`
  and `VF_FIT` say. (The second translation at launch takes no kernel with products to the third
  level: such a kernel keeps the pairs of its first translation, or its large block is refused,
  with safe mode or without.) In the PTX lowering a pair gives the bits of each product by itself
  (`docs/pairs.md`, section 13): those pairs stay. Measured on 2026-11-24 on one V100, in three
  rounds alternated after a warm-up: the fp16 GEMMs of Triton told 8.0 (PTX) and of cuBLAS ran at
  the same speed with `VF_SAFE=1` as without, Triton's with the same bits, while PyTorch's fp16
  attention (`scaled_dot_product_attention`, sm_75 machine code whose products pair) took 5.24 ms a
  call instead of 2.80, 1.9 times as long.
- **It is read once.** `libcuda.so.1` reads it when it is loaded, the nvJitLink adapter and the
  bf16 cuBLAS variants at their first use: a program that sets `VF_SAFE` while it runs does not get
  safe mode from `libcuda.so.1`, and may get it from the others.

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
- **The last bits of a Triton FP16 product can depend on `num_stages`** (Triton told 8.0, whose
  `tl.dot` reaches the layer as PTX). A product whose two operands come straight from plain
  `ldmatrix` is added on the halves of k (two by two on one A where they pair, with the same
  bits), one whose operands are gathered from other threads on the even, then the odd k
  (`docs/emulations.md`, "Operands loaded from shared memory"). In a pipelined kernel
  (`num_stages` 2 or more) the operands of the first products of each turn of the loop are
  register copies of what the turn before loaded, which the layer gathers; with one stage none
  are. With B stored K x N (`ldmatrix.trans`), every product adds the even, then the odd k,
  whatever the stages. Both orders are valid sums
  of the same terms, by Volta's instruction four terms at a time; neither is wrong. Measured on
  2026-11-23 on one V100, Triton 3.8 through the layer, float16 inputs and outputs, float32
  accumulators, on 2048 x 7344 x 1536, 256 x 151936 x 1536 and 4096 x 4096 x 4096, with and
  without `VF_PAIRS`: with B stored N x K (K contiguous, as a linear layer's weight), the three
  1-stage configurations (tiles 128 x 128 x 32 with 8 or 4 warps, 64 x 128 x 32) gave one result,
  which was cuBLAS 12.9's bit for bit; the two 3-stage ones (128 x 128 x 32, 64 x 64 x 32)
  another, and 128 x 128 x 64 with 2 stages a third, each differing from cuBLAS's in 0.09 to
  0.15 % of the outputs, by at most 0.03125. With B stored K x N, all six configurations gave the
  same bits (0.16 to 0.20 % of the outputs off cuBLAS's).
- **Conversions of the 4-bit and 8-bit formats**: identical to an RTX 5090's on every value of
  the formats.
- **Block-scaled products**: see the table above.
