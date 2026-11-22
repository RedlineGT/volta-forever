# volta-forever: known limits

What does not work through the layer, what is slower, and what was not tried. Everything was
measured on one Tesla V100 with driver 580.178.04 and no switch set; a figure carries its date
where it is not from the build of this release. `docs/` says how each figure was obtained:
`docs/validation.md` for what was compared with what, `docs/performance.md` for the timings,
`docs/emulations.md` for what is computed in place of the instructions that a V100 lacks.

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

A kernel that the layer cannot run is refused when the program asks for it: the program gets an
error that names the kernel, and a line on standard error says why. The other kernels of its
module load and run.

**PyTorch 2.14.0.** Of the 23,693 functions for sm_75 in the 475 units of machine code of its
library, 23,663 are translated (99.9 %) and 30 are refused: 10 for a branch through a uniform
register, 6 for the instruction `I2IP`, 4 for a call whose return address is not given the
usual way, 4 for a conversion of a uniform register, 2 for `IMAD.HI.X`, 2 for a jump table, 2
for want of registers. None of them is reached by the tests below.

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
the translator does not have (`LEA.HI` with a uniform register).

**Other libraries.** The machine code of 42 libraries of CUDA 13.5, cuDNN, NCCL and others was
translated and checked instruction by instruction, not run (`docs/validation.md`, "The corpora"):
of 3,108 units, 3,105 are translated in whole or in part; three are not read at all (a section
table that points outside the file; two sections of code of the same name). In these libraries
more functions are refused than in PyTorch (10,986 of 70,978); cuFFT's kernels for double
precision are among them (relocations that the translator cannot carry with their instructions).
The CUDA 13 libraries are not meant to be translated: the layer puts the CUDA 12.9 builds, which
have code for Volta, behind their names (`INSTALL.md`).

**Cubins for sm_80 and later.** VF can use the PTX that NVIDIA's assembler retains in
`.nv_debug_ptx_txt` with line information. It checks the architecture and the kernel names
before compiling that PTX for Volta. This covers standalone cubins from Triton and cubins in
fat binaries. For sm_75, VF also tries retained PTX when its default machine-code translation
refuses every kernel in the image.

Builds from `main` also translate a subset of stripped `sm_80` executables. VF checks each function's
instruction forms, branches and memory addressing. Global loads and stores, including 128-bit
accesses, require the CUDA system descriptor on every path. VF lowers the supported 16-byte
global-to-shared copy to a guarded Volta load and store and waits for the completed group.
SGLang uses this path for its top-eight expert sum and its wheel’s 32-bit expert counting and
alignment kernels. The router uses shared counter increments and GPU-scope 32-bit global atomic adds.
FlashInfer uses the same translator for 2D RMSNorm, fused residual RMSNorm and 3D query/key
normalization. For compiler-generated warp helpers, VF checks the call target, return register
and descriptor writes. Unsupported functions are refused individually. Other stripped targets above
`sm_75`, including `sm_86`, still require compatible PTX or a second supported image.

**PTX.** A kernel that uses an instruction with no equivalent on Volta is refused: clusters of
blocks, Hopper's barriers, tensor maps (`docs/emulations.md`, "Other instructions").
For an indirect call within a single PTX module, an explicit `.calltargets` list identifies the
possible callees. With only a `.callprototype`, VF follows each device function that
could match the declared input and return lists. It distinguishes ordinary 32-bit and
64-bit scalars. Unfamiliar declarations, including array parameters, leave all targets
possible. A kernel is refused if one of those paths reaches a trapped function.
When linking separate device objects, an indirect call specified only by a prototype
can still be refused if any linked object contains a trapped device function.

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

## CUDA graphs

A Tesla V100 can capture and replay translated kernels. The official SGLang AWQ server used CUDA graphs for both prefill and decode. A graph can report error 701 when it is instantiated or replayed if the translated kernel needs too many registers for the recorded block size. For translated machine code, `VF_BORROW=9 vf …` can preserve a source block size that would otherwise be lost. Shared-memory limits still apply. [SGLang measurements](docs/guides/benchmarks/sglang-default-operators.json) · [GPU validation](docs/validation.md#cuda-graph-replay)

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

- Running translated compute kernels concurrently on multiple devices within one process
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
