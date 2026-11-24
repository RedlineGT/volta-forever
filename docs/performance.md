# Performance: what the layer costs

Every figure here is from one Tesla V100-SXM2-32GB (compute capability 7.0) with driver
580.178.04, and carries its date: the layer changed between measurements, and a figure is from
the build of this release only where it says so. "Native" is defined for each comparison.

## 1. Method

### The machine

One Tesla V100-SXM2-32GB: 80 multiprocessors, each with 65,536 registers and at most 2,048
threads. NVIDIA driver 580.178.04 (the last branch with Volta support). Programs built with CUDA
13.5.0 run through the layer; the CUDA 12.9.2 libraries stand behind the forwarding libraries.

### What "native" means

- **PyTorch operations**: PyTorch 2.14.0 built for CUDA 12.6 (`torch==2.14.0` from
  `download.pytorch.org/whl/cu126`), the last wheel that carries machine code for sm_70. The
  same version of PyTorch, the same operations, the same GPU, without the layer. The two wheels
  are run one after the other on one GPU by `tests/gpu/torch-bench.sh`.
- **llama.cpp**: a build of the same sources (b12418) for sm_70 with CUDA 12.9.2, run without
  the layer.
- **Hand-written Volta kernels** (section 6): kernels of 1Cat-vLLM
  (`https://github.com/1CatAI/1Cat-vLLM`, commit `6db9c816cd`, its `main` branch of 2026-11-02),
  lifted into a standalone test program and run on the same V100. They are not the same
  computation as anything the layer runs: they are what Volta can do when the code is written
  for it.

There is no build of vLLM, SGLang or ZML for sm_70 to compare with. Their speeds are given
without a ratio. (Tried for vLLM 0.34.0 on 2026-11-21, with PyTorch 2.14.0 for CUDA 12.6 put
in the place of its PyTorch: the engine does not start. Its own compiled extension is linked
with CUDA 13's runtime library; with that library where it finds it, its torchvision refuses
a PyTorch built for another CUDA.)

### How timings are taken

- **PyTorch operations** (`tests/gpu/torch-bench.py`): each operation is called several times,
  the GPU waited for after each call (`torch.cuda.synchronize`); the median time of a call is
  kept. A number that depends on the result is written with it, so that the two wheels can be
  seen to have computed the same thing.
- **llama.cpp**: its own `llama-bench` (three repetitions) and, for single operations, its
  backend test `test-backend-ops perf`, both builds in one session.
- **Engines** (vLLM, SGLang, ZML): tokens per second as the engine reports them, one small model
  on one GPU. These are end-to-end figures, not kernel timings.
- **The profile** (`VF_PROFILE`, `crates/vf-proxy/src/profile.rs`): every launch is followed
  by a wait for its stream, and the time counted is from the return of the launch to the return
  of that wait. This puts the launches one after the other and holds the CPU: a profile says
  where the time is, it is not a benchmark.
- **Shares of a kernel's time** come from CUPTI's sampling of the program counter, used through
  the layer; the translator's map says what each instruction of a translation is for.
- **Loops and single instructions**: test kernels written for the purpose, three runs, on a GPU
  held for the whole comparison.
- **The clock**: after the GPU has been idle, it runs below its boost clock (at 1290, then
  1425 MHz) for up to about ten seconds of work, and a kernel whose time follows the clock takes up
  to 19 % longer meanwhile (cuBLAS 12.9's bf16 product with a bias, 1536 x 2 x 1536, through
  PyTorch on 2026-11-23: 247.7 us at 1290 MHz, 212 us at 1530 MHz). Compare timings taken after
  the clock has settled (the GPU kept busy until NVML reports 1530 MHz), or record the clock with
  each.

### Measuring on a V100

For a measurement of one's own, beside the warm-up of "The clock" above (12 s of work was enough
for the figures below):

- **Alternate the settings inside one job**: A, B, A, B on one GPU in one job, each after the
  warm-up, rather than one job for each setting. From one pass to the next of one session, a kernel
  of the same bytes ran at 26.0 to 27.9 TFLOPS (Triton's float16 GEMM, 2026-11-23): more than many
  of the differences that this file reports.
- **Time small kernels through a CUDA graph**, not one eager launch at a time from Python: there a
  launch costs the launcher's microseconds (Triton's more than PyTorch's), and a launch waited for
  costs the wait too. Through the layer at 1530 MHz (2026-11-23, PyTorch 2.14.0 and Triton 3.8.0,
  two rounds): an addition of 4,194,304 floats took 64 us a launch replayed from a graph of 200
  launches, and 65 us launched 200 times back to back, but 82 to 87 us (PyTorch) and 97 to 104 us
  (Triton) with a wait after each launch; one of 4,096 floats took 1.4 us from the graph, 7 to 10
  us (PyTorch) and 18 to 20 us (Triton) back to back, and 18 to 86 us with a wait after each. A
  kernel of tens of microseconds timed one eager launch at a time measures the launcher as much
  as the kernel.

## 2. PyTorch operations

89 operations, the layer's time divided by the native wheel's (`tests/gpu/torch-bench.sh`),
in three runs of 2026-11-20 with the build of that day, two of them in one session:

| | |
|---|---|
| Range | 0.88 to 1.65, 0.89 to 1.64, 0.89 to 1.65 |
| Median | 1.00 in each |
| Geometric mean | 1.03 in each (1.026, 1.025, 1.028) |
| Within 3 % of the native time | 69, 72 and 71 of the 89 |

Two more runs, on 2026-11-21 with the builds of that day (the proxy changed in between, the
translator did not): 0.79 to 1.65 in both, median 1.00, geometric mean 1.02, 71 and 72 of the
89 within 3 %, and the same operations over 1.10 as below, `cumsum` at 1.10. The 0.79 is `add`
on bytes: the native wheel took 0.073 ms for it in both runs of that day, where it had taken
about what the layer takes the day before; the layer's 0.057 ms is the faster of its two
times, below.

Those that take over 1.10 times the native time in one of the three runs:

| Operation | Layer / native in the three runs | Cause |
|---|---:|---|
| `attention.f16` | 1.65, 1.64, 1.65 | Turing's tensor-core products are emulated (section 5) |
| `std.dim` | 1.34, 1.30, 1.35 | registers: one block fewer on each multiprocessor (below) |
| `argmax.dim`, `max.dim` | 1.24 to 1.25, 1.21 to 1.23 | the same |
| `cumsum`, `randn` | 1.24, 1.10, 1.24; 1.12 to 1.16 | registers |
| `lib.conv2d`, `model.cnn` | 1.22 to 1.23, 1.20 to 1.24 | the algorithm that the newer cuDNN picks, not the layer (below) |
| `model.transformer_layer.f16`, `attention` | 1.12 | |

Two are faster than the native wheel by more than 3 % in every run (`adaptive_avg_pool2d` 0.88
to 0.89, `layer_norm` 0.95): the two wheels are the same PyTorch built with different compilers
and libraries. For each operation the test also compares the sums of the two results: all 89
agree, in each run.

**From one run to the next.** A few operations take one of two times, the same build and the
same translated code giving the one or the other for a whole run: `cumsum` 0.307 or 0.348 ms
(1.10 or 1.24 times the native time), `std.dim` 0.140 or 0.146 ms, the small CNN 4.32 or
4.44 ms, `add` on bytes 0.057 or 0.062 ms. The native wheel's times do not move so (`cumsum`:
0.279 to 0.285 ms in 43 runs). The layer says the same things in a slower run as in a faster
one (its log is the same, the times apart), and the cause is not known. It is not the build:
in 43 runs of that day, 22 with a build that differs from this one in the translation of 10 of
PyTorch's 475 cubins (none of them the cubins of these kernels) and 21 with this one, `cumsum`
was at its slower time in 7 and in 4, `std.dim` in 9 and in 7; and in a series of 24 runs
where the two builds took turns, `cumsum` was at its slower time in every fourth run, six
times, whichever build that was. Over the 89 operations the two builds are not told apart: in
one session, two runs each in turn, the layer's times with this build are 0.999 and 1.002
times those with the other (geometric means).

Two causes were separated on 2026-11-17 with the CUDA 12.6 wheel, which holds code for both
sm_70 and sm_75: the same wheel run with its sm_70 code, then with its sm_75 code translated
(`VF_TRANSLATE=force`). The reductions, `cumsum` and `randn` were slower by the same amounts;
the convolution was not.

1. **Registers and occupancy.** The compiler for sm_75 keeps loop bounds and strides in uniform
   registers; the translation gives each of them an ordinary register, and the kernel's count
   rises. A V100 gives a thread its registers in groups of eight, and a multiprocessor has
   65,536: when the count crosses a multiple of eight, one block fewer may fit. PyTorch's
   reduction kernels for `max` and `argmax` have 32 registers in the sm_70 build, 36 in the
   sm_75 build and 40 once translated: with blocks of 512 threads, three blocks on a
   multiprocessor instead of four. `std` goes from 31 to 37, the same step; `sum` goes from 28
   to 31, stays at four blocks and is not slower. The step alone, measured by raising the
   register count of the native `max` kernel from 32 to 40: 24 % more time (0.091 to 0.113 ms).
2. **cuDNN.** The CUDA 13 wheel carries cuDNN 9.24, the CUDA 12.6 wheel cuDNN 9.10; the newer
   one picks a slower algorithm for the same 3x3 convolution. With the older cuDNN put in front
   of the CUDA 13 wheel's (through the audit library's map), `conv2d` went from 1.25 times the
   native time to 1.00, and the small CNN from 1.21 to 1.00.

## 3. Language-model engines

### llama.cpp

The official build for CUDA 13.5 (b12418) through the layer, against the build of the same
sources for sm_70.

**A small model** (Qwen2.5-0.5B; `tests/gpu/llamacpp-bench.sh`: `llama-bench`, three
repetitions, every layer on the GPU, the two builds in one session, 2026-11-20; tokens per
second):

| Weights | Flash attention | Prompt of 2,048 tokens: layer | native | Generation of 128 tokens: layer | native |
|---|---|---:|---:|---:|---:|
| 4 bits (Q4_0) | off | 10,398 | 19,557 | 437 | 430 |
| 4 bits (Q4_0) | on | 11,922 | 25,793 | 472 | 482 |
| 8 bits (Q8_0) | off | 9,715 | 20,814 | 410 | 402 |
| 8 bits (Q8_0) | on | 11,104 | 27,774 | 436 | 447 |

Generation, one token at a time, is at the native build's speed (97.5 to 102 %): its kernels
have nothing that is emulated. A prompt is processed at 40 to 53 % of the native build's speed
(prompts of 512 tokens: 38 to 56 %, with more spread from one repetition to the next). For a
batch the two builds do not run the same code: the official build, told capability 7.5, takes
its kernels with integer tensor-core products, which the layer computes with integer
instructions, and its tensor-core attention; the native build leaves batches of 64 tokens or
more to cuBLAS (section 4 has the same comparison on a large model, where it goes the other
way).

**Flash attention**, 196 cases of llama.cpp's own backend test (2026-11-18):

| Head size | Cases | Native total (ms) | Layer / native |
|---|---:|---:|---:|
| 40 | 1 | 9 | 0.59 |
| 64 | 17 | 489 | 1.42 |
| 128 | 17 | 692 | 1.63 |
| 256 | 26 | 2,378 | 1.27 |
| 512 | 10 | 6,814 | 1.26 |
| 576 | 12 | 6,106 | 2.13 |
| **all 196** | | **16,538** | **1.60** |

More than half of these cases compare two different kernels. llama.cpp chooses its attention
kernel by the device: on a Volta it takes a kernel without tensor cores for small batches,
while the official build, told that it runs on a Turing, takes its tensor-core kernel. In 86 of
the 196 cases both builds run the tensor-core kernel; over those the ratio was 1.71 (quartiles
1.56, 1.78, 1.93; worst case 2.37), measured before a change to the lowering of chains of
products and not computed again since.

In those like-for-like cases the layer runs as many Volta products as the native code. What it
adds is the gathering of the B operands that the program loads with `ldmatrix.trans` (five
shuffles for a register) and the integer arithmetic around it (`docs/emulations.md`, "Matrix
loads and moves"). By samples of the program counter on one case (head size 256, batch 512):
the products 29 % of the time, the integer arithmetic of the emulation 29 %, the stores of key
and value tiles to shared memory 17 %, the loads from it 13 %, the shuffles 5 %.

### vLLM

vLLM 0.34.0 installed with pip, Qwen2.5-0.5B-Instruct in f16, three prompts of 50 tokens, one
at a time (runs of 2026-11-20 and 21): greedy generation at 84 to 99 tokens per second, 83 to
97 with sampling. The same build gives 85 or 96 from one run to the next. The engine's text is
the CPU's token for token (`docs/validation.md`, "On the hardware").

### SGLang

SGLang 0.5.23 installed with pip, Qwen2.5-0.5B-Instruct in f16, the same three prompts, one
at a time (runs of 2026-11-20 and 21): 45 to 52 tokens per second with its default attention
(FlashInfer), 55 to 59 with `--attention-backend triton`.

### ZML

The upstream build, Qwen3.5-9B in f16 (`tests/gpu/zml.sh`; runs of 2026-11-19 to 21): 28.8 to
29.4 tokens per second when the program is told capability 7.5, 27.4 to 28.2 when told 8.0.
The layer of those days told 8.0 to the build's cuDNN as well. Since 0.1.3 the cuDNN is told
7.5 when 8.0 is asked, because its kernels run from the translation of their sm_75 machine
code; `libzml_cuda.so`, whose kernels are PTX, is still told 8.0 (`docs/architecture.md`, "What
a program is told about the GPU").

## 4. NVFP4 weights

Qwen3.8-27B with NVFP4 weights (27 billion parameters; 19.6 GB as a GGUF file, converted by
llama.cpp's own script), with the official llama.cpp build b12418 (`llama-bench`, three
repetitions, the three builds in one session, 2026-11-20):

| Build | Prompt of 512 tokens (tokens/s) | Generation of 128 tokens (tokens/s) |
|---|---:|---:|
| The official build through the layer | 299 | 34.5 |
| The native sm_70 build as it is | 224 | 35.0 |
| The native sm_70 build with its quantized kernels forced (`GGML_CUDA_FORCE_MMQ`) | 582 | 35.0 |

On GPUs before Blackwell, llama.cpp computes NVFP4 weights with integer products on 8-bit
activations. For one token at a time its kernel uses `dp4a`, which a V100 has: nothing is
emulated, and generation is at 98.5 % of the native build's speed.

For a batch, the official build, told capability 7.5, takes its kernels with integer tensor-core
products. A V100 has no such product, and the layer computes each one with integer
instructions (`docs/emulations.md`, "Integer products"): **1.95 times the time of the same
kernels compiled for Volta** (the third row, where the build is made to use its quantized
kernels, which on Volta are written with `dp4a`). The native build as it is does something else
for batches of 64 tokens or more: it leaves them to cuBLAS, with the weights made floats again
for each batch; the layer's emulated kernels are 1.33 times faster than that.

The 48 tokens generated through the layer were the CPU's, and the mean KL divergence from the
CPU's next-token distributions 0.0004 with flash attention and 0.0005 without (2026-11-20, when
the first two rows were also measured again: 300 and 34.6 through the layer, 224 and 34.9
native).

## 5. Where the time goes

### What instructions cost on a V100

Measured with loops written for the purpose (2026-11-18):

- **A full multiprocessor executes about 4.4 instructions of a warp per nanosecond, whatever
  they are** (5 or 100 arithmetic instructions a turn, or 26 of which 8 are shuffles): with its
  64 warps, each gets one instruction every 14.5 ns. A shuffle is one instruction like another
  when the multiprocessor is full.
- **It does 0.37 loads from shared memory per nanosecond**, of one, two or four words alike (with
  the addresses tried, four words to a bank), and already 0.27 with two warps: a load costs
  eleven times an instruction's share. A kernel with such loads waits for them, and its
  arithmetic costs nothing more.
- **It does 0.7 Volta tensor-core products per nanosecond**, with half-precision accumulators
  (two steps) as with float ones (four steps).
- **A warp with little company waits for its own chains**: with two warps on a multiprocessor,
  an instruction whose result the next one needs costs 2.8 ns for arithmetic, 3 to 5 ns for a
  product, 20 ns for a shuffle.

So what an emulation costs depends on the kernel around it: in a full multiprocessor, on the
count of instructions it adds; in a kernel of few warps, on the shuffles it puts in a chain.

### A Turing tensor-core product

A V100's product is 8x8x4 on a group of eight threads; Turing's is 16x8x16 on the 32 threads of
a warp, with the operands laid out otherwise. In machine code, one Turing product by itself
becomes two Volta products (eight steps), with the elements of B gathered from other threads
by shuffles and each thread's results chosen afterwards: a function with one such product and
float accumulators has 33 instructions more than the original, 39 with half-precision
accumulators (the counts that `tests/gpu/run.sh` holds the translator to).

Where a program loads its operands as whole rows and computes two products on the same A, the
translator computes the two together on rows loaded directly in Volta's layout, without the
shuffles for B (`docs/translator.md`, "Tensor-core products in machine code"). Measured in a
loop of eight Turing products a turn (2026-11-18; milliseconds for 80, 160 and 640 blocks of
2,000 turns, that is one, two and eight blocks on each multiprocessor):

| The loop | Instructions a turn | 80 blocks | 160 | 640 |
|---|---:|---:|---:|---:|
| Translated, each product by itself | 247 | 2.00 | 2.35 | 8.39 |
| Translated, products two by two on whole rows | | 0.48 | 0.63 | 2.05 |
| Written by hand in that arrangement | 68 | 0.34 | 0.49 | 2.08 |
| Written by hand for Volta (the floor) | 54 | 0.27 | 0.36 | about 1.5 |

With eight blocks on a multiprocessor the translation is where the loop written by hand in the
same arrangement is, 1.4 times the floor. With one block it is 1.43 times slower than the loop
by hand, and why is not understood.

On a whole kernel, PyTorch's attention in half precision
(`fmha_cutlassF_f16_aligned_64x64_rf`: 64 products; batch 64, 12 heads, 512 tokens, head size
64), by the profile (2026-11-18):

| | Each product by itself | Two by two on rows | Native sm_70 build |
|---|---:|---:|---:|
| Instructions | 4,803 | 3,732 | 3,576 |
| Registers | 219 | 238 | 168 |
| Blocks on a multiprocessor | 2 | 2 | 3 |
| Time of a launch (ms) | 5.19 | 3.05 | 1.66 |

(The sm_75 code that is translated has 3,016 instructions.) The benchmark of section 2, which
does not put launches one after the other, gave 1.64 for this operation on 2026-11-20.

Where the 3.05 ms go, by samples of the program counter (68 million samples):

| Instructions | Share of the samples |
|---|---:|
| Unchanged from the original | 28 % |
| The product steps | 16 % |
| Loads of whole rows from shared memory | 13 % |
| Addresses of those rows (shuffles, additions), and the waits for the loads | 17 % |
| Accumulators converted between the two layouts | 5 % |
| Uniform registers made ordinary | 4 % |
| The rest of what is added | 17 % |

Part of the distance to the native build is occupancy and not instructions: with its register
count raised from 168 to 176, the native kernel itself goes from three blocks to two and from
1.66 to 1.86 ms (12 %), and no further up to 208.

### Triton's matrix products

Triton 3.8 writes tensor-core instructions for `tl.dot` only for a GPU it is told is 8.0 or
later. Told 7.5, it falls back to float32 `fma`, and says so when asked
(`MLIR_ENABLE_DIAGNOSTICS=remarks`: "Dot op using MMA for compute capability 75 has been
deprecated. It falls back to the FMA path."). In a program with PyTorch, Triton is told what
PyTorch's CUDA runtime is told: 7.5 by default and with `VF_CAPABILITY=8.0` alone, 8.0 with
`VF_CAPABILITY_ALL=1` (`LIMITS.md`, "The capability told with `VF_CAPABILITY`").

The same kernels told 7.5 and told 8.0, through the layer, in one session (2026-11-23, one V100,
Triton 3.8.0 and PyTorch 2.14.0 of a vLLM 0.34.0 installation; TFLOPS from the median of three
batches of launches; every output compared with a float64 computation on the CPU, and as close
to it told 7.5 as told 8.0, except the float32 GEMM, which Triton computes in tf32 told 8.0):

| Kernel | Told 7.5 | Told 8.0 |
|---|---:|---:|
| float16 GEMM 2048x7344x1536, tiles 128x128x32, 8 warps, 1 stage | 0.23 | 33.1 |
| the same, 3 stages | 0.23 | 27.2 |
| the same, tiles 64x64x32, 4 warps, 3 stages | 3.25 | 25.6 |
| bf16 GEMM, tiles 128x128x32, 3 stages (told 8.0: the scalar bf16 products, `docs/emulations.md`) | 0.23 | 10.1 |
| float32 GEMM, tiles 64x64x32 (told 8.0: tf32) | 6.14 | 7.62 |
| vLLM's `fused_moe_kernel`, float16, 256 tokens (E=128, top-8, K=2048, N=1536; vLLM's default tiles) | 0.095 | 4.38 |
| the same, 2,048 tokens | 0.153 | 12.9 |
| the same, bf16, 256 and 2,048 tokens | 0.097, 0.154 | 2.35, 6.86 |
| vLLM's `kernel_unified_attention`, float16, prefill of 4 x 1,024 tokens (32 heads, 8 KV heads, size 128) | 2.57 | 13.6 |
| the same, decode of 64 sequences of 2,048 tokens | 0.54 | 1.84 |

Where the time goes told 7.5, for the first GEMM: Triton's sm_75 cubin, as NVIDIA's ptxas makes
it from Triton's PTX, has 64 registers and 3,952 bytes of stack, and 5,952 instructions of which
2,048 `FFMA`, 2,056 local loads and 1,151 local stores: the 64 accumulators of a thread and the
operands do not fit, and ptxas keeps the kernel at 64 registers. The layer's translation of it has
5,963 instructions (11 added), 66 registers and the same stack and local accesses. The PTX
compiled for sm_70 directly by the same ptxas is worse: 7,800 instructions, 3,298 local loads and
1,791 stores, 32 registers. The slowness is Triton's code for 7.5, not its translation; and no
translation of a sum of scalar `fma` can reach the tensor cores, whose sums round otherwise.

Told 8.0, Triton's cubins keep their PTX, which the layer lowers to Volta's products ("Pairs of
products" in `docs/emulations.md`).

### Registers

A translation needs registers of its own: one for each uniform register of the function, two
to compute into, more for a tensor-core product. By default they are taken above the
function's own, which is the fastest code and raises the count. On PyTorch's 475 cubins
(2026-11-19), of the 22,909 kernels that this translates by themselves, 6,131 end in a higher
group of eight registers than they declared, and 1,012 go from 64 or fewer to more than 64.

Keeping every kernel within its declared count is possible (registers of the function that
hold nothing are borrowed; failing that the uniform registers live in the thread's local
memory: `docs/translator.md`, "Registers") and was measured on the 89 operations of section 2,
each pair in one session: geometric mean 1.026 for the usual translation against 1.032 with
registers borrowed, and 1.002 against 1.024 with local memory used as well. `std` along a
dimension gains (1.22 to 1.04: it gets its fourth block back), `unique` loses 20 %, `cumsum`
14 %, `layer_norm` 12 %, `softmax` 8 %: a borrowed register that must be saved and brought back
costs more than a register above. So the layer does not do it for kernels that translate the
usual way. It does it in two cases only:

- for a kernel that would otherwise be refused because it names nearly every register of the
  machine (what `VF_BORROW=7`, the default, adds: the code of every other kernel is the same,
  byte for byte, as without it);
- for a kernel that the driver refuses to launch (below).

### A launch that the driver refuses

A kernel compiled for blocks of 1,024 threads may have 64 registers at most
(`docs/architecture.md`, "Launch"). When the usual translation has taken it over that, the
driver refuses the launch, and the layer translates the kernel's module again within each
kernel's declared count and launches from there. In PyTorch's test battery this happens for
`avg_pool2d` and `interpolate` in float32 and float64 and for `unfold` in float32, and in three
of 32 small whole models.

The refused launch costs 1.0 microsecond before the second one is made (200,000 refused
launches, 0.99 to 1.04 microseconds each, 2026-11-20). The second translation is slower code;
how much slower than a native build was not measured.

### The first use

The first time a program runs through the layer, work is done that later runs do not repeat.
For llama.cpp (12 modules of 14 to 20 MB of PTX; 2026-11-16): 3.1 s to rewrite the PTX for
Volta, 2.5 s to work out what each kernel depends on, and the driver's own compilation of the
kernels that are used (1.1 to 1.8 s for each of the large attention kernels when the driver's
cache of compiled kernels is empty too).

The layer keeps what it worked out in a cache (`VF_PROXY_CACHE`, `~/.cache/volta-forever/proxy`
by default; `docs/architecture.md`, "The path of a module"):

| | Cache empty | Cache filled |
|---|---:|---:|
| llama.cpp, 48 tokens generated (2026-11-16) | 6.9 s | 1.3 s |
| PyTorch, a battery of 820 operations (2026-11-17) | 63 s | 5 to 7 s |

### A block-scaled product

One of Blackwell's block-scaled products (`docs/emulations.md`, "Block-scaled products")
becomes 485 instructions: 64 tensor-core steps, 66 shuffles, 50 selections, and the decoding
of the 4-bit values and the arithmetic on the scales (counted on the rewritten text compiled for
sm_70, 2026-11-18). The tensor-core steps are not at their minimum: 32 would make the same
sums, but each block goes through the plain product's sequence, which uses half of each Volta
instruction. Section 6 has the time.

## 6. The target: kernels written by hand for Volta

What a layer that rewrites one instruction at a time can reach is bounded by code written for
Volta from the start. Two kernels of 1Cat-vLLM (section 1) were lifted into a standalone test
program and timed on the same V100 (2026-11-18, three runs, the median).

### An NVFP4 matrix-vector product (34,816 x 5,120)

| Kernel | Time (microseconds) | Weights read (GB/s) |
|---|---:|---:|
| 1Cat-vLLM's kernel, written by hand for Volta | 138 | 726 |
| A kernel written with Blackwell's block-scaled product, through the layer | 374 | 268 |

2.7 times. The hand-written kernel keeps the weights packed in the order that Volta's product
reads them: for sixteen elements of a row it runs 29 instructions (16 tensor-core steps, 9
half-precision multiplications that decode the 4-bit values and apply the scale, 4 loads) and
no shuffle. The emulation cannot repack anything: it replaces one instruction of a program
whose data it knows nothing of, and must bring each block's values to the threads where
Volta's product wants them.

### Attention

1Cat-vLLM's attention kernel for head size 256 (24 query heads, 4 key-value heads, causal, half
precision), written with 16x16x16 products that the sm_70 compiler makes Volta products of:

| Chunk | Key-value length | Time (ms) | TFLOPS |
|---:|---:|---:|---:|
| 2,048 | 2,048 | 1.35 | 38.3 |
| 2,048 | 8,192 | 8.21 | 44.0 |
| 2,048 | 32,768 | 35.4 | 45.2 |
| 2,048 | 65,536 | 68.8 | 47.2 |

Its 1,840 instructions hold 512 tensor-core steps and 20 shuffles, all of them in the
reductions of the softmax: the operands are loaded from shared memory in the layout that the
product expects. No program runs this shape through the layer, so there is no figure of the
layer's to put beside it; the closest are the two of section 5 (PyTorch's attention at 1.64 to
1.84 times its own native build) and of section 3 (llama.cpp's at 1.71 in the like-for-like
cases).

### What is left on the table

Three things separate translated attention from code written for Volta, in the order of the
profile of section 5: operands that the program loads in Turing's layout and that must be
loaded again or gathered in Volta's; the registers that uniform registers take once they are
ordinary ones, which cost a block of occupancy; and the instructions around each emulated
matrix load. Products two by two on whole rows removed most of the first where the program's
loads allow it. A program that packs its data for Volta, as 1Cat-vLLM's kernels do, does not
go through any of this: it uses Volta's instructions, and the layer leaves its code as it is.

## 7. bf16 products through cuBLAS: the opt-in variants on real programs

Measured on 2026-11-23 on one Tesla V100-SXM2-32GB, driver 580.178.04, with the layer built from
the series of the bf16 cuBLAS variants before its last four fixes (the signed zero with a bias, the
attributes of an explicit cuBLASLt algorithm, workspaces not at 256-byte addresses, the lazy setup;
none changes the speed of a product that a path serves). PyTorch 2.14.0+cu130 and transformers
5.22.2, eager, in bf16: Qwen2.5-0.5B-Instruct (hidden size 896, MLP 4,864, vocabulary 151,936) for
generation and training, BAAI/bge-base-en-v1.5 for embeddings, openai/whisper-small for speech.
Each comparison is one GPU job in which the three settings (both switches off,
`VF_CUBLAS_BF16_SKINNY=1`, `VF_CUBLAS_BF16_TC=1`, which turns the skinny path on too) take turns
over two rounds, each process after 12 s of warm-up so that the card's clock has settled (1,500 to
1,530 MHz recorded). The cases bound by the host (generation at batch 1 and 8, Whisper's
transcription) ran pinned to the CPUs nearest the GPU and moved by up to about 30 % from one start
of the process to the next. The rate with the switches off, then the ratios to it, and the share of the
GPU's kernel time that bf16 cuBLAS products take with the switches off:

| Workload | Switches off | Skinny | Tensor cores | bf16 products |
|---|---:|---:|---:|---:|
| `generate()`, batch 1 (one column: never served) | 35 tokens/s | 1.00 | 0.97-1.02 | 35 % |
| `generate()`, batch 8 | 196 tokens/s | 1.27 | 1.35 | 82 % |
| `generate()`, batch 32 | 635 tokens/s | 1.00 | 1.46-1.49 | 80 % |
| Prompts of 512 to 2,048 tokens | 7.4k-10.4k tokens/s | 1.00 | 1.31-1.70 | |
| Log-likelihood of 8 x 512 tokens | 8.8k tokens/s | 1.00 | 1.56 | |
| The same model under `torch.compile`, at `VF_CAPABILITY=8.0 VF_CAPABILITY_ALL=1` (one round) | 8.5k-12k tokens/s | | 1.44-1.91 | |
| Embeddings, batches of 32 to 256 | 91-1,600 sequences/s | 1.00 | 1.51-1.74 | 82.5 % |
| Whisper's encoder alone | 21 windows/s | 1.00 | 1.30 | 48 % (Whisper) |
| Whisper's transcription, batch 16 | 103 audio-seconds/s | 1.04 | 1.24 | |
| LoRA training step, 8 x 256 and 32 x 128 tokens | 1.40 and 0.81 steps/s | 1.02, 1.04 | 0.81, 0.56 | 83 % |
| Full training step, two sizes | 1.23 and 0.70 steps/s | 1.00 | 0.66, 0.37 | 77.5 % |

Training is slower with the tensor-core path because the products of its backward pass fall into
the path's scalar fallback: the input gradients (NN, m = 896, n = the tokens, k = 4,864) ran at 0.20
times the speed of cuBLAS 12.9, the weight gradients (NT) at 0.16 to 0.33 times and LoRA's rank-16
products at 0.16 to 0.26 times; in the profile of full training, the fallback kernel took 31.7 % of
the kernel time, the scaling kernel about 34 % and the fp16 products 12.5 %. With the skinny path
alone, every output was bit for bit that with the switches off, in every workload and round, the
losses and final weights of training included. With the tensor-core path, generation's logits
differed from those with the switches off by 0.048 on average and 0.92 at most, with the same most
likely token at 98.05 % of positions; from those of the same model in fp32 on the CPU, by 0.0602 on
average with the switches off and 0.0600 with the path (same most likely token: 97.66 % and
97.27 %); embeddings had a cosine of at least 0.999947 to those with the switches off, and their
nearest neighbour was the CPU's in every case, with the path and without.

An engine that replays its steps from CUDA graphs gains little (`LIMITS.md`, "The bf16 cuBLAS
variants"): SGLang at its default settings computes in fp16 on this card, with no bf16 product, and
made to use bf16 it replays prefill and decode from graphs. Not measured: other models of each
kind, another card, engines other than those named here.
