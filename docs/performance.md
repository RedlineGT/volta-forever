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

Measured on 2026-11-24 on one Tesla V100-SXM2-32GB, driver 580.178.04, with the layer of the series of the bf16
variants' rules of speed: the tensor-core path's gate on k (`VF_TC_MIN_K`, 1024), its three rules of speed (below),
its fallback launched once per call with its tiles chosen from measured times. PyTorch 2.14.0+cu130 and
transformers 5.22.2, eager, in bf16: Qwen2.5-0.5B-Instruct (hidden size 896, MLP 4,864, vocabulary 151,936) for
generation and training, BAAI/bge-base-en-v1.5 for embeddings, openai/whisper-small for speech. Each comparison is
one GPU job in which the three settings (both switches off, `VF_CUBLAS_BF16_SKINNY=1`, `VF_CUBLAS_BF16_TC=1`, which
turns the skinny path on too) take turns over two rounds, each process after 12 s of warm-up so that the card's
clock has settled (1,530 MHz recorded), every process pinned to the CPUs nearest the GPU (the cases bound by the
host, generation at batch 1 and 8 and Whisper's transcription, moved by up to about 30 % from one start of the
process to the next unpinned). The rate with the switches off, then the ratios to it (the range over the two
rounds), and the share of the GPU's kernel time that bf16 cuBLAS products take with the switches off (from the
profiles of the earlier measurements, 2026-11-23: what runs with the switches off is not changed by the series):

| Workload | Switches off | Skinny | Tensor cores | bf16 products |
|---|---:|---:|---:|---:|
| `generate()`, batch 1 (one column: never served) | 36 tokens/s | 0.98 | 0.98-0.99 | 35 % |
| `generate()`, batch 8 | 197 tokens/s | 1.27 | 1.39 | 82 % |
| `generate()`, batch 32 | 637 tokens/s | 1.00 | 1.46 | 80 % |
| Prompts of 512 to 2,048 tokens | 7.4k-10.4k tokens/s | 1.00 | 1.05-1.18 | |
| Log-likelihood of 8 x 512 tokens | 8.8k tokens/s | 1.00 | 1.06 | |
| The same model under `torch.compile`, at `VF_CAPABILITY=8.0 VF_CAPABILITY_ALL=1` (one round) | 8.6k-12k tokens/s | | 1.07-1.22 | |
| Embeddings, batches of 32 to 256 | 91-1,600 sequences/s | 1.00 | 1.11-1.16 | 82.5 % |
| Whisper's encoder alone | 21 windows/s | 1.00 | 1.07 | 48 % (Whisper) |
| Whisper's transcription, batch 16 | 104 audio-seconds/s | 1.04 | 1.20-1.21 | |
| LoRA training step, 8 x 256 and 32 x 128 tokens | 1.40 and 0.81 steps/s | 1.03, 1.03 | 1.07, 1.07 | 83 % |
| Full training step, 8 x 256 and 32 x 128 tokens | 1.24 and 0.70 steps/s | 1.00 | 1.02, 1.02 | 77.5 % |

With the gate on k, every product of these three models with 768 or 896 in-features, all but the second product of
each MLP (3,072 or 4,864 in-features), goes to the skinny path or to cuBLAS 12.9: only one of the seven products of
each layer of the generation model, and one of the six of each layer of the two encoders, can go to the tensor
cores. The prompts, the scoring, the embeddings and Whisper's encoder therefore gain less than in the measurements
made before the gate, when the path served their products with 768 or 896 in-features too: 1.31 to 1.70, 1.56,
1.51 to 1.74 and 1.30 then. Generation at batch 8 and 32 kept its gain (1.35 and 1.46 to 1.49 then).

With the skinny path alone, every output was bit for bit that with the switches off, in every workload and round:
the generated tokens and logits, the embeddings, Whisper's transcripts, and the losses and trained weights of
training. With the tensor-core path, the logits of a check batch (4 x 64 positions) differed from those with the
switches off by 0.047 on average and 0.81 at most, with the same most likely token at 98.44 % of positions; from
those of the same model in fp32 on the CPU, by 0.0602 on average with the switches off and 0.0584 with the path
(the same most likely token at 97.66 % of positions with both). Greedy generation at batch 8 kept the tokens of the
switches off for 77 of its 128 on average (one sequence of the 8 throughout), at batch 32 for 38; at batch 8 no
sequence was the CPU's throughout with the switches off (42 tokens on average), 2 of 8 with the path (59).
Embeddings had a cosine of at least 0.999952 to those with the switches off, and their nearest neighbour was the
CPU's in every case, with the path and without. Of Whisper's 24 transcripts at batch 8 and 16, 18 were those of
the switches off; of the others, two differ in a capital, two in "21," against "21st,", one in a comma and a word,
and in one window the switches off stopped after the first word where the path gave the sentence. The losses of
training differed from those with the switches off by 0.0042 at most; the first three of LoRA were within 0.0019
of the CPU's in fp32 with the switches off and within 0.0017 with the path. Each setting gave the same bits in both
rounds, in every workload.

### Training

bf16 fine-tuning of the 0.5B model: adapters of rank 16 on every linear layer (LoRA; the base frozen in bf16)
and full fine-tuning (fp32 master weights), both under bf16 autocast with AdamW, 8 steps of 8 x 256 and of
32 x 128 tokens after 2 of warm-up, on fixed windows of the model's own README and licence. Measured again in a
job of its own, three runs of each setting alternated; beside them, two runs of the layer as it was before this
series (the path then served every layout), from an earlier job of the same kind. Steps per second:

| Workload | Switches off | Tensor cores | Ratio | Tensor cores, before this series | Ratio |
|---|---:|---:|---:|---:|---:|
| LoRA, 8 x 256 tokens | 1.405 | 1.508 | 1.07 | 1.088 | 0.77 |
| LoRA, 32 x 128 tokens | 0.808 | 0.863 | 1.07 | 0.444 | 0.55 |
| Full fine-tuning, 8 x 256 tokens | 1.246 | 1.271 | 1.02 | 0.777 | 0.62 |
| Full fine-tuning, 32 x 128 tokens | 0.702 | 0.717 | 1.02 | 0.250 | 0.36 |

Every bf16 product of a real step timed in place (each call waited for; three runs of each setting), summed per
step; then with the layout rule removed, by the test builds' controls `VF_BF16_CONTROL=tc-layouts,tc-tall,tc-wide`
(no product of these steps is tall or wide), which serve the NN and NT products again:

| Workload | Switches off (ms per step) | Tensor cores | Speed | Layout rule removed | Speed |
|---|---:|---:|---:|---:|---:|
| LoRA, 8 x 256 tokens | 643.2 | 604.7 | 1.06 | 764.4 | 0.84 |
| LoRA, 32 x 128 tokens | 1,097.6 | 1,019.7 | 1.08 | 1,402.0 | 0.78 |
| Full fine-tuning, 8 x 256 tokens | 651.0 | 636.6 | 1.02 | 897.1 | 0.73 |
| Full fine-tuning, 32 x 128 tokens | 1,211.0 | 1,181.3 | 1.03 | 1,976.9 | 0.61 |

With the rule removed, 20 of the 22 NN and NT products within the path's gate (k from 1,024 to 16,384, n of 32
or more) ran at 0.24 to 0.92 times 12.9's speed: the output layer's weight gradient (NT 896 x 151,936 x 4,096) at
0.29, the MLP's input gradients (NN 896 x n x 4,864) at 0.42 to 0.44, a rank-16 adapter's B gradient (NT 16 x 4,864
x 4,096) at 0.24; the other two, of 16 rows, at 1.01 and 1.35. The TN products that the path serves ran at 1.45 to
3.88 times 12.9's speed: the down projections (896 x n x 4,864) at 1.45 to 1.53, the adapters' 16 x n x 4,864 at
2.12 and 3.88. In each setting every step's loss and the trained weights' checksum were the same bits from run to
run; with the path they differ from those with the switches off (the losses by 0.0042 at most), since its products
differ from 12.9's in their last bits. With the rules by class (no product of these steps is in a class they
changed), the same steps measured again in a job of their own, through the release runtime, three runs of each
setting alternated: steps per second at 1.07 to 1.08 and 1.07 times those with the switches off for LoRA (8 x 256
and 32 x 128 tokens), 1.02 and 1.02 for full fine-tuning (the range over the three pairs of runs); every product of
the steps timed in place, summed per step, at 1.02 to 1.08 times.

### The rules of speed

The tensor-core path serves a class of shapes when it has real members, layers of real models measured with their
own operands, and over every layer and n measured in it its median speed is 1.10 times cuBLAS 12.9's or more and its
slowest 0.90 or more, and no model measured has its products' summed time above 12.9's at any n, in inference or in
training: a sweep on operands that convert exactly does not open a class. The classes are regions of shape, the same
for every model, those that a rule carves or could carve: the tiers of the tall rule (m >= 8,192 with up to 64
columns, and with 65 to 128), the long products of many rows (m >= 6,144 with k > 9,216 and up to 256 columns), the
tiers of the wide rule (k > 8,192 with 257 to 320 columns, and with more), the layouts other than TN, and the
products that no rule covers, of which the one product measured below the floor owes it to its data, not to its
shape (the exception, below). Three static rules of shape, decided before the gate on k, leave the classes that fail
to 12.9, and with them the products of those classes at which the path was faster: each gives its reason in the exit
report, and a control of the test builds removes it (`VF_BF16_CONTROL`), with which the rule's tests fail
(`bf16-rules`, `tests/forward/bf16.sh`). Speeds are 12.9's time over the path's. The classes, on the real operands
of eight models (below: every linear layer of the gate, n = 32 to 2,048, two choices of columns, the path measured
on every TN product, 3 rounds), on the products of four fine-tuning steps timed in place ("Training", above, with
the layout rule removed) and on the sweeps of the long products of many rows and of the tall rule's n bound of 64
(B15), below (operands that convert exactly; the earlier sweeps below are not in the table):

| Class | Real members | Median | The slowest (its input's columns failing) | Points of the sweeps | Median | The slowest | Criterion | Rule |
|---|---|---:|---|---:|---:|---|---|---|
| Served remainder: every other TN product of the gate | 18,636 products of 8 models | 2.22 | 1.09: the MLP's down projection of the 0.5B model, 896 x 1,024 x 4,864 (the first n tokens; 12 % failing) | 214 | 1.44 | 1.05: 7,680 x 128 x 9,216 | passes | served |
| Tall, n <= 64: TN, m >= 8,192 | 1,506 products of 3 models | 1.07 | 0.66: the output layer of the 4B hybrid, 248,320 x 64 x 2,560 (n spread over the batch; none failing) | 64 | 1.02 | 0.70: 248,320 x 64 x 4,096 | fails: median 1.07 < 1.10, slowest 0.66 < 0.90 | to 12.9 |
| Tall, 65 to 128 columns: TN, m >= 8,192 | 1,004 products of 3 models | 1.24 | 1.04: the output layer of the 1.5B model, 151,936 x 128 x 1,536 (the first n tokens; none failing) | 148 | 1.37 | 1.05: 10,240 x 112 x 4,096 | passes | served |
| Long products of many rows: TN, m >= 6,144, k > 9,216, n <= 256 | none |  |  | 951 | 1.30 | 0.94: 6,656 x 192 x 13,824 | fails: no real members | to 12.9 |
| Wide, 257 to 320 columns: TN, k > 8,192 | 184 products of 3 models | 1.60 | 0.85: the MLP's down projection of the 1.5B model, 1,536 x 320 x 8,960 (the first n tokens; 64 % failing) |  |  |  | fails: slowest 0.85 < 0.90 | to 12.9 |
| Wide, more than 320 columns: TN, k > 8,192 | 1,196 products of 3 models | 1.39 | 0.62: the MLP's down projection of the 1.5B model, 1,536 x 768 x 8,960 (n spread over the batch; 55 % failing) |  |  |  | fails: slowest 0.62 < 0.90 | to 12.9 |
| The exception, in the served remainder: one layer, the 9B hybrid's first linear-attention output projection (TN 4,096 x n x 4,096, 20 to 38 % of its input's columns failing) | 29 products of 1 model | 1.19 | 0.88 at n = 320 (30 %; across processes; 0.857 in one process, at n = 320) | | | | below the floor; no rule of shape tells it from its shape's other products | served |
| Layouts other than TN: NN, NT, TT (a fine-tuning step's backward products, timed in place) | 22 products of 1 model | 0.51 | 0.24: NT 16 x 4,864 x 4,096 | | | | fails: median 0.51 < 1.10, slowest 0.24 < 0.90 | to 12.9 |

The wide rule's class of 257 to 320 columns fails on its slowest product alone (its median: 1.60): the 1.5B
model's MLP down projection with the most failing columns, TN 1,536 x 320 x 8,960 at n = 320 (the first 320
tokens of the prefill), 64 % of its input's columns failing, at 0.85; with the 320 columns spread over the batch
the same layer ran at 1.08, and the class's 182 other products (the down projections of the 1.5B model, the 9B
hybrid and the 4B hybrid at n = 320) at 1.05 to 2.15. How many of a layer's columns fail depends on its data,
which no rule of shape sees: the class goes to 12.9 whole, and those gains with it.

The long products of many rows have no real member: no layer of the eight models has m >= 6,144 with k > 9,216 and up
to 256 columns. On operands that convert exactly they ran at a median of 1.30 and at 0.94 at worst (the sweeps below);
the class goes to 12.9 with the tall rule's long-k branch, as before the classes. Its likely members, the down
projections of SiLU-gated MLPs of hidden size 6,144 or more with an MLP of 9,217 to 16,384, have the inputs whose
columns fail the exact check most (the wide rule's), and with 30 to 60 % of their input's columns failing such shapes
ran at 0.52 to 0.89 times 12.9's speed (measured in review, on synthetic operands: TN 7,168 x 128 x 12,288 at 0.52
with 60 % failing, 6,144 x 128 x 16,384 at 0.70 and 0.61 with 30 and 60 %).

With these rules, each model's summed step (the sum of its products' times over the layers measured, as a speed
against 12.9) at its lowest over every n and both choices of columns, the exception's layer served, against the
rules before the classes (the tall rule then took 65 to 128 columns too) and every product served; the fine-tuning
steps' products together ran at 1.02 to 1.08 times 12.9's speed, and at 1.02 to 1.08 times in the check under these
rules ("Training", above):

| Model | Lowest summed step with these rules (n, choice of columns) | With the rules before | Every product served |
|---|---|---|---|
| 0.5B model | 1.495 (n = 2,048, the first n tokens) | 1.495 | 1.495 |
| 1.5B model | 1.521 (n = 320, the first n tokens) | 1.521 | 1.760 |
| 9B hybrid | 1.488 (n = 320, n spread over the batch) | 1.308 | 1.543 |
| 4B hybrid | 1.577 (n = 320, n spread over the batch) | 1.549 | 1.693 |
| T5-base | 1.730 (n = 1,536, n spread over the batch) | 1.730 | 1.730 |
| OPT-125m | 1.732 (n = 1,536, the first n tokens) | 1.732 | 1.732 |
| bge-base-en-v1.5 | 1.716 (n = 1,536, the first n tokens) | 1.716 | 1.716 |
| whisper-small's encoder | 2.137 (n = 1,024, the first n tokens) | 2.137 | 2.137 |

The rules:

| Rule | Products left to 12.9 | Reason in the exit report | Measured on the side it leaves to 12.9 | Measured on the side it serves | Control |
|---|---|---|---|---|---|
| Layout | every layout but TN (op(A) = A^T, op(B) = B): a backward pass's input gradients (NN) and weight gradients (NT), TT | `not TN (the tensor-core path's layout)` | the NN and NT products of four fine-tuning steps, timed in place with the rule removed: 20 of the 22 that the path would serve at 0.24 to 0.92 (the output layer's weight gradient, NT 896 x 151,936 x 4,096, at 0.29), the two others, of 16 rows, at 1.01 and 1.35; the steps' products together at 0.61 to 0.84 | the same steps' TN products: 1.45 to 3.88; the steps' products together at 1.02 to 1.08 | `tc-layouts` |
| Tall | TN with m >= 8,192 and n <= 64, or with m >= 6,144 and k > 9,216 | `m >= 8192 with n <= 64, or m >= 6144 with k > 9216 (12.9 as fast at some of these shapes)` | output layers (m = 151,936 and 248,320, k = 1,024 to 4,096) at n <= 64: 0.65 to 0.96; at k = 1,536 and n = 64, 1.05 at m = 8,192, 1.02 to 1.03 at 8,320 to 8,448, 0.95 to 1.00 from 8,704; the sweep B15 (m = 8,192 to 248,320, k = 1,024 to 4,096) at n = 56 and 64: a median of 0.92, 28 of 42 points below 1.0, down to 0.70 (248,320 x 64 x 4,096); with m >= 6,144, k > 9,216 and n <= 256: no layer of a real model; on the sweeps 48 of 951 points below 1.03, down to 0.94, in bands that follow 12.9's waves, and the other widths of the same shapes at up to 1.9 | output layers at n = 65 to 128: 1.01 to 1.56, from 129: 1.21 and more; B15 at 65 to 128 columns: 126 points, a median of 1.40, the slowest 1.05 (10,240 x 112 x 4,096); with k up to 9,216 at m >= 6,144 and n <= 256: 1.05 and more (sweeps below) | `tc-tall` |
| Wide | TN with k > 8,192 and n > 256 | `k > 8192 with n > 256 (the fallback's cost on real activations)` | a down projection (k = 8,960) with 53 to 70 % of its input's columns failing, the rule removed by its control: 0.85 and 1.08 at n = 320 (the two choices of columns), 0.62 to 0.98 at every n from 384 to 2,048 | the same layer at n = 256: 1.15 to 1.17, at n = 192: 1.17 to 1.23 | `tc-wide` |
| None: the exception | nothing (no rule of shape tells these products from others of their shape) | none; from `VF_LOG_LEVEL=2` the exit report counts the calls with more than a quarter of their rows or columns failing | | served, and can run below 12.9's speed: a product whose input has a quarter or more of its columns failing the exact check, near the top of one of 12.9's waves; measured on one layer, TN 4,096 x n x 4,096 with 26 to 38 % failing: 0.86 to 0.97 at n = 128, 192 and 256 to 320 (below) | |

The classes and the bounds come from the sweeps below and from the real operands of four language models (a 0.5B
model of hidden size 896, a 1.5B model of hidden size 1,536, and two hybrid models of linear-attention and attention
layers, 9B of hidden size 4,096 and 4B of hidden size 2,560): every linear layer with k >= 1,024 of one prefill of 8
x 256 tokens of text, at n = 32 to 2,048 in 15 steps, two choices of n of those columns (the first n tokens; n
spread over the batch), 3 rounds, the settings alternated in separate processes, the path measured on every TN
product of its gate (a test build with the tall and wide rules removed by their controls). The products the rules
serve:

| Model | Layers | Products served | Median speed | The slowest served | Below 1.05 | Below 1.0 | Products left to 12.9 |
|---|---:|---:|---:|---|---:|---:|---:|
| 0.5B, hidden 896 | 24 | 696 | 3.28 | 1.09: the MLP's down projection, 896 x 4,864, n = 1,024 (12 % of its columns failing) | 0 | 0 | 0 |
| 1.5B, hidden 1,536 | 197 | 4,951 | 2.56 | 1.04: the output layer, 151,936 x 1,536, n = 128 (none of its columns failing) | 2 | 0 | 762 |
| 9B hybrid, hidden 4,096 | 249 | 6,159 | 1.95 | 0.88: a linear-attention output projection, 4,096 x 4,096, n = 320 (30 % of its columns failing) | 9 | 6 | 1,062 |
| 4B hybrid, hidden 2,560 | 249 | 6,159 | 2.06 | 1.04: the output layer, 248,320 x 2,560, n = 128 (none of its columns failing) | 2 | 0 | 1,062 |

and four models with ReLU and GELU activations (their MLPs' second products, k = 3,072, are their only layers of
k >= 1,024; no column of their inputs failed the check):

| Model | Layers | Products served | Median speed | The slowest served | Products left to 12.9 |
|---|---:|---:|---:|---|---:|
| T5-base (ReLU), encoder and decoder | 24 | 696 | 3.92 | 1.71: 768 x 3,072, n = 1,536 | 0 |
| OPT-125m (ReLU) | 12 | 348 | 3.92 | 1.70: 768 x 3,072, n = 1,536 | 0 |
| bge-base-en-v1.5 (GELU, BERT encoder) | 12 | 348 | 3.90 | 1.67: 768 x 3,072, n = 1,536 | 0 |
| whisper-small's encoder (GELU, 1,500 frames of audio) | 12 | 312 | 4.58 | 2.11: 768 x 3,072, n = 1,024 | 0 |

Every product the rules serve in these eight models ran at 1.04 times 12.9's speed or more, but in one layer: the
exception below.

**The exception: a product whose input has many failing columns.** 12.9 computes a bf16 product in tiles of 64 x 64,
four per SM, 320 at a time on a V100: its time is flat over ranges of n and jumps where one more of its tiles lands
on the busiest SM (at m = 4,096: at n = 129, 193, and 321, where its second wave begins). The path's own time grows
evenly with n, and its fallback adds one fused multiply-add chain over the whole k for every element of a failing
column, less efficient per column than 12.9's tiles. With a quarter or more of the columns failing, the sum can pass
12.9's time on its flat stretches, most before its second wave. The one layer measured with such an input, the
output projection of the first linear-attention layer of the 9B model (TN 4,096 x n x 4,096, 26 to 38 % of its
input's columns failing; in the other 248 layers measured in that model, at most one column in 32 failed), on that
layer's own operands, 12.9 and the path alternated in one process per n (8 event-timed calls per setting, 7 rounds,
medians; 2 passes; the worse of the two choices of columns):

| n | 12.9's tiles (at most per SM) | 12.9 (us) | The path (us) | Speed (two passes) | The attention layer's speed |
|---:|---|---:|---:|---|---:|
| 32 | 64 (1) | 632 | 440 | 1.435, 1.438 | 2.41 |
| 64 | 64 (1) | 635 | 527 | 1.204, 1.206 | 2.15 |
| 96 | 128 (2) | 667 | 648 | 1.028, 1.028 | 1.56 |
| 128 | 128 (2) | 670 | 700 | 0.956, 0.955 | 1.73 |
| 160 | 192 (3) | 741 | 739 | 1.003, 1.003 | 1.74 |
| 192 | 192 (3) | 746 | 767 | 0.972, 0.973 | 1.65 |
| 224 | 256 (4) | 931 | 797 | 1.168, 1.170 | 1.93 |
| 256 | 256 (4) | 935 | 995 | 0.939, 0.940 | 1.73 |
| 288 | 320 (4) | 918 | 1,052 | 0.873, 0.874 | 1.33 |
| 320 | 320 (4) | 922 | 1,075 | 0.857, 0.856 | 1.47 |
| 336 | 384 (2 waves) | 1,564 | 1,084 | 1.442, 1.440 | 2.46 |
| 384 | 384 (2 waves) | 1,572 | 1,309 | 1.201, 1.200 | 2.23 |
| 448 | 448 (2 waves) | 1,642 | 1,394 | 1.178, 1.178 | 2.08 |
| 512 | 512 (2 waves) | 1,830 | 1,633 | 1.120, 1.120 | 2.15 |
| 576 | 576 (2 waves) | 1,833 | 1,745 | 1.050, 1.050 | 1.75 |
| 640 | 640 (2 waves) | 1,826 | 1,774 | 1.029, 1.030 | 1.68 |
| 656 | 704 (3 waves) | 2,471 | 1,854 | 1.332, 1.333 | 2.11 |
| 704 | 704 (3 waves) | 2,479 | 1,892 | 1.310, 1.311 | 2.21 |
| 768 | 768 (3 waves) | 2,548 | 1,997 | 1.276, 1.278 | 2.05 |
| 1,024 | 1,024 (4 waves) | 3,382 | 2,867 | 1.180, 1.180 | 1.98 |

The runs of the models above, settings alternated in separate processes, give the same picture for that layer (0.88
to 1.13 at n = 96 to 320, 1.15 and more from 384 on); an earlier run of that shape alone gave higher figures (0.92
at n = 288 to 320): between separate processes the card's clock can differ, and 12.9's arithmetic-bound kernel
follows it more than the path's memory-bound steps, which is why the table above was measured in one process. Before
the fallback's tiles were chosen from measured times, the runs across processes gave the layer 0.88 to 1.02 at n =
96 to 320 as well (0.90 at n = 128, 0.95 since). Its effect on the model is small: it takes 0.2 % of the 12.9 time
of the model's layers measured, and their sum ran at 1.31 to 1.77 times 12.9's speed at n = 96 to 320; at n = 320,
1.488 times, against 1.489 had that layer run at 12.9's speed. Three ways out were measured and not taken. A choice
of the internal fp16 GEMM's algorithm gains at most 3 % of a call: cuBLASLt's heuristic gives seven, its first is
the fastest up to n = 104 and within 1.3 to 6.7 % at n = 128 to 336, and the step at n = 321 is 12.9's, not the
path's (the path's time is flat there: 1,075 against 1,084 us). Handing such a call to 12.9 whole needs its failing
count on the host, a synchronization per tensor-core call, which cost 10.9 % of the speed of eager decoding at batch
32 (1.1 % on prompts of 512 tokens, 0.2 % on 2,048), is impossible in a graph capture, and would make an element's
bits depend on the other columns' data. A rule of shape would take the layer's attention-layer twins too (the same
shape, no failing column: 1.33 to 2.46). From `VF_LOG_LEVEL=2` the exit report counts the calls with more than a
quarter of their rows or columns failing.

**Failing columns at long k.** The exception is one real layer, but the cost it shows is its data's, and at long k it
is larger: the fallback adds one chain over the whole k for each element of a failing column. Measured in review, on
synthetic operands (a failing column made by one element at 2^-40; a release runtime of the rules by class that also
served the long products of many rows, every point served; 12.9 and the path alternated in one process, 12 s of
warm-up, 3 rounds of 8 event-timed calls), the speed against 12.9 (the median of the rounds):

| Shape (TN m x n x k) | Class | Every column converting | 30 % failing | 60 % failing |
|---|---|---:|---:|---:|
| 4,096 x 256 x 12,288 (the 9B hybrid's down projection) | served remainder | 1.39 | 0.83 | 0.74 |
| 4,096 x 128 x 12,288 | served remainder | | | 0.77 |
| 5,120 x 256 x 13,824 (hidden 5,120, MLP 13,824) | served remainder | 0.98 | 0.67 | 0.55 |
| 5,120 x 128 x 13,824 | served remainder | | | 0.70 |
| 2,560 x 256 x 9,216 (the 4B hybrid's down projection) | served remainder | | | 0.84 |
| 1,536 x 256 x 8,960 (the 1.5B model's down projection) | served remainder | | | 1.18 |
| 4,096 x 320 x 4,096 (the exception's shape) | served remainder | 1.45 | 0.85 | |
| 6,144 x 128 x 16,384 (hidden 6,144, MLP 16,384) | long products of many rows (to 12.9) | 1.19 | 0.70 | 0.61 |
| 6,144 x 256 x 16,384 | long products of many rows (to 12.9) | 1.38 | 0.89 | 0.85 |
| 7,168 x 128 x 12,288 | long products of many rows (to 12.9) | 1.07 | 0.66 | 0.52 |
| 6,656 x 192 x 13,824 | long products of many rows (to 12.9) | 0.94 | 0.65 | 0.53 |

Where the path's margin with converting inputs is wide (2.2 to 2.6 times 12.9's speed at 1,536 x 256 x 8,960) it stays
faster with 60 % of the columns failing; where it is moderate (0.94 to 1.4), 30 % failing takes it to 0.65 to 0.89 and
60 % to 0.52 to 0.85: about half of 12.9's speed at the slowest, and more failing columns cost more. The real layers
measured at long k had few failing columns: at most 2 % of the 9B hybrid's down projections' (k = 12,288), 3 % of the
4B hybrid's (k = 9,216); the 1.5B model's one layer with 53 to 70 % (k = 8,960) is in the wide classes, which go to
12.9 from 257 columns. So the floor of 0.90 holds for the products measured, not for every product of a served class:
a program whose inputs fail the check at long k can run below it, and from `VF_LOG_LEVEL=2` the exit report counts its
calls with more than a quarter of their rows or columns failing (in review it counted each of the 338 synthetic
calls).

What each rule costs and keeps, per model, from the same runs: the path measured on every TN
product of its gate (a test build with the tall and wide rules removed by their controls, which leave the path's
computation as it is) against 12.9 in separate processes, 3 rounds, and a product that a rule leaves to 12.9 counted
at 12.9's time. The summed step is the sum of the products' times over the layers measured (the first n tokens), as
a speed against 12.9; the tables of every n from 32 to 2,048, with both choices of columns, are in this section's
evidence:

| Model | n | Layers | 12.9, sum (ms) | Every product served | With the rules | Without the tall rule | Without the wide rule | Left to 12.9: tall / wide |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5B | 64 | 24 | 18.96 | 6.33 | 6.33 | 6.33 | 6.33 | 0 / 0 |
| 0.5B | 128 | 24 | 18.62 | 5.52 | 5.52 | 5.52 | 5.52 | 0 / 0 |
| 0.5B | 512 | 24 | 18.97 | 2.51 | 2.51 | 2.51 | 2.51 | 0 / 0 |
| 0.5B | 2,048 | 24 | 45.60 | 1.49 | 1.49 | 1.49 | 1.49 | 0 / 0 |
| 1.5B | 64 | 197 | 86.36 | 2.28 | 2.36 | 2.28 | 2.36 | 57 / 0 |
| 1.5B | 128 | 197 | 92.88 | 2.11 | 2.11 | 2.11 | 2.11 | 0 / 0 |
| 1.5B | 512 | 197 | 183.31 | 1.89 | 1.59 | 1.59 | 1.89 | 0 / 28 |
| 1.5B | 2,048 | 197 | 604.50 | 1.76 | 1.55 | 1.55 | 1.76 | 0 / 28 |
| 9B hybrid | 64 | 249 | 227.95 | 1.54 | 1.56 | 1.54 | 1.56 | 97 / 0 |
| 9B hybrid | 128 | 249 | 307.18 | 1.66 | 1.66 | 1.66 | 1.66 | 0 / 0 |
| 9B hybrid | 512 | 249 | 845.92 | 1.84 | 1.66 | 1.66 | 1.84 | 0 / 32 |
| 9B hybrid | 2,048 | 249 | 3,018.98 | 1.71 | 1.61 | 1.61 | 1.71 | 0 / 32 |
| 4B hybrid | 64 | 249 | 155.87 | 1.78 | 1.82 | 1.78 | 1.82 | 97 / 0 |
| 4B hybrid | 128 | 249 | 179.45 | 1.71 | 1.71 | 1.71 | 1.71 | 0 / 0 |
| 4B hybrid | 512 | 249 | 438.09 | 1.85 | 1.74 | 1.74 | 1.85 | 0 / 32 |
| 4B hybrid | 2,048 | 249 | 1,579.80 | 1.81 | 1.69 | 1.69 | 1.81 | 0 / 32 |

The four ReLU and GELU models have no product that a rule leaves to 12.9: their summed step is the same with and
without the rules. The tall rule, at n <= 64, gives 12.9 the MLPs' up projections and the output layers of the three
larger models, where 12.9 was as fast or faster at many shapes: with it their summed step is a little faster than
with every product served (at n = 64: 2.36 against 2.28 on the 1.5B model, 1.56 against 1.54 on the 9B hybrid, 1.82
against 1.78 on the 4B hybrid); from 65 columns on it gives 12.9 nothing (at n = 128 the rules before the classes,
which gave it that tier too, ran the 9B hybrid at 1.31 against 1.66 now, the 4B hybrid at 1.56 against 1.71 and the
1.5B model at 1.98 against 2.11). The wide rule, from n = 320 on, gives 12.9 their MLPs' down projections (1.89
against 1.59 on the 1.5B model, 1.84 against 1.66 on the 9B hybrid, 1.85 against 1.74 on the 4B hybrid at n = 512,
every product served against the rules): its class's slowest products, 0.62 times 12.9's speed on real text, fail
the criterion, and its faster ones go to 12.9 with them. The tall rule's long-k branch costs these models nothing, as
none of them has a product there.

The tier from 65 to 128 columns of the tall rule (the MLPs' up projections and the output layers of the three larger
models at a batch of 65 to 128 tokens) passes the criterion: on these models it ran at 1.24 (median) and 1.04 at
worst; the earlier sweep of the n bound (k = 1,536 to 7,168, below) found points at 112 to 128 columns, at m =
10,240 and on output layers, where 12.9 was within 3 % of the path or faster (down to 0.95), above the floor, and
B15 (k = 1,024 to 4,096) none below 1.05. Through the release runtime of these rules, on the same models' layers of
m >= 8,192 (n = 32 to 192, the first n tokens and n spread over the batch, against 12.9 in separate processes, 3
rounds): at n = 96 and 128, the tier these rules serve, 1,004 products at a median of 1.24, the slowest 1.04 (the
output layer of the 1.5B model, 151,936 x 128 x 1,536, the first n tokens), up to 1.76; at n = 192, served before
too, 502 products at a median of 1.63, the slowest 1.25; at n <= 64, which the rule leaves to 12.9, 1,506 products at
1.00 (12.9 against itself: 0.97 to 1.04).

**The sweeps** (synthetic operands, every row and column converting, TN; the medians of 2 or 3 passes):

- the tall rule's m bound, n = 32 to 128, k = 1,536 to 4,096 and 10,240 to 12,288, m = 8,192 to 9,208: at
  k = 1,536 and 64 columns the path's lead fell from 1.05 at m = 8,192 to 1.02 at 8,384 and below 1.0 from 8,704
  (0.95 at 9,208); at 65 to 128 columns it stayed at 1.07 to 1.46; at k = 10,240 to 12,288 it was within 2 % of
  12.9 from m = 8,256 on.
- the n bound, m = 8,192 to 151,936, k = 1,536 to 7,168, n = 96 to 192: 0.95 to 1.70 at n = 96 to 128 (within 3 %
  of 12.9 or below at 112 to 128 columns for m = 10,240 and the output layers, down to 0.95), 1.18 and more at
  n = 136 to 192.
- output layers, m = 151,936 and 248,320, k = 1,024 to 4,096: 0.65 to 0.96 at n <= 64, 1.01 to 1.56 at n = 65 to
  128, 1.21 and more from 129.
- the tall rule's n bound of 64 (sweep B15; 3 passes, the tall and wide rules removed by their controls): m = 8,192
  to 248,320, k = 1,024 to 4,096, n = 56 to 128 (168 points): at n = 56 and 64 a median of 0.92, 28 of 42 points
  below 1.0, down to 0.70 (248,320 x 64 x 4,096); at 65 to 128 columns a median of 1.40, none of 126 below 1.0, the
  slowest 1.05 (10,240 x 112 x 4,096), up to 1.93.
- the tall rule's long-k branch (TN with m >= 6,144 and k > 9,216): k = 8,448 to 13,824 at m = 6,144 to 8,192 in
  steps of 128 to 256 and n = 48 to 256; k = 14,336 to 16,384 at m = 6,144 and 6,272; k = 12,288 to 16,384 at m =
  8,320 to 32,768 and n = 160 to 256; and k = 8,448 to 9,216 at m = 8,320 to 32,768 (3 passes each, the rule removed
  by its control), with the earlier corner (k = 12,288 to 16,384, m = 6,144 to 8,320, n = 48 to 384). With k up to
  9,216, every point ran at 1.05 or more (214 points; the slowest, 7,680 x 128 x 9,216, at 1.05). Past k = 9,216, 48
  of 951 points were below 1.03, down to 0.94: at 128 columns for m = 6,912 to 7,680 from k = 10,240 on, at 192
  columns for m = 6,144 to 6,656 from k = 12,288 on (at m = 6,144 only at k = 16,384, 1.03) and for m = 20,480 to
  32,768 from k = 13,824 on, at 64 and 96 columns for m = 7,680 to 8,064 from k = 12,800 on, and at 256 columns for
  m = 10,240 to 20,480 from k = 13,824 on; the other widths of the same shapes ran at up to 1.9 (160 and 224 columns
  at 1.07 and more everywhere; the median of the 951 points, 1.30). These bands follow 12.9's waves: from 96 columns
  on, every point below 1.03 was one where 12.9's 64 x 64 output tiles, ceil(m / 64) ceil(n / 64) of them dealt to
  the 80 SMs, fill at least 90 % of the places that the busiest SM's count gives all of them, so that 12.9 wastes
  almost nothing of its last wave; most such points were still faster (44 of 495 were not), and at 48 to 64 columns
  the four points below 1.03 do not follow it (a fill of 0.78 to 0.79). No bound of shape follows the bands, so the
  branch takes the region, the faster widths with it; no layer of the eight models measured has such a shape, and a
  class without a real member goes to 12.9.
- the wide rule's n bound on real operands (the runs above, the rule removed by its control): the down projection
  with the most failing columns (k = 8,960) at 1.15 to 1.23 at n = 192 to 256, at 0.85 at n = 320 (the first
  tokens; 1.08 with the columns spread over the batch) and below 1.0 at every n from 384 on (down to 0.62); no
  model measured has a down projection of k between 4,864 and 8,960; those of k = 4,864 kept at least 1.08 times
  12.9's speed at every n.

### Where the GPU time goes

Profiles with `VF_PROFILE` (every launch waited for: a profile, not a benchmark), with the tensor-core path on: the
share of the GPU's kernel time taken by cuBLAS 12.9's bf16 kernels, by the path's fp16 products, by the layer's
other kernels (the skinny kernel, the scaling and check, the epilogue) and by the fallback; then, from the report at
exit, the path's calls that had rows or columns failing the exact check, and the share of the path's elements that
the fallback computed:

| Program | 12.9's bf16 kernels | The path's fp16 products | The layer's other kernels | The fallback | Calls with failing rows or columns | Elements by the fallback |
|---|---:|---:|---:|---:|---:|---:|
| LoRA training, 2 steps of 8 x 256 tokens | 75.2 % | 2.4 % | 4.1 % | 0.3 % | 16 of 192 | 2.82 % |
| Full fine-tuning, 2 steps of 8 x 256 tokens | 73.0 % | 1.8 % | 1.9 % | 0.2 % | 25 of 96 | 2.84 % |
| A prompt of 2,048 tokens | 42.7 % | 5.3 % | 5.8 % | 0.6 % | 16 of 72 | 2.10 % |
| Log-likelihood of 8 x 512 tokens | 66.8 % | 5.5 % | 5.6 % | 0.7 % | 16 of 72 | 3.14 % |
| `generate()`, batch 8 | 35.4 % | 3.7 % | 15.4 % | 1.3 % | 142 of 3,288 | 2.24 % |
| `generate()`, batch 32 | 62.3 % | 2.3 % | 4.0 % | 0.7 % | 160 of 3,288 | 2.36 % |

With the switches off, 12.9's bf16 kernels took 57 to 83 % of the kernel time of the same programs. With the path
on, they still take most of it: the products with 896 in-features and, in training, the backward pass's. Before
this series, in full fine-tuning, the fallback took 31.7 % of the kernel time, the scaling kernel about 34 % and
the fp16 products 12.5 %, for the backward pass's products that now go to 12.9.

### Verify mode on these programs

With `VF_CUBLAS_BF16_VERIFY=1` (the release runtime), each call the skinny path served was made a second time
through cuBLAS 12.9 and the two outputs compared bit for bit on the GPU. Generation at batch 2 to 16 and prompts of
512 tokens: 111,744 calls verified, in 8 call signatures; LoRA fine-tuning at batches of 16 tokens: 12,606, in 5;
SGLang serving the generation model in bf16 (at `VF_CAPABILITY=8.0 VF_CAPABILITY_ALL=1`, as its guide says),
1 to 128 requests generating 128 tokens each and 1 to 32 prompts of 512 tokens, three rounds, its decode replayed
from CUDA graphs:
74,208 calls verified, in 13 call signatures, of which 1,584 were made outside a graph and the others were replays
of captured calls. None differed, and the same calls verified again with the tensor-core path on did not either. With the tensor-core path on as well, the fallback's elements of the path's calls
that had any were compared the same way: those of 1,601 calls (14.6 million elements) in generation and of 137
calls (0.57 million) in fine-tuning, and of 200 calls (158 million) in SGLang's prefills; none differed. The embedding encoder at batches of 1 to 16 tokens
gave the skinny path no call: cuBLAS 12.9 does not run its sequential kernel on its shapes there. Verify mode is a
diagnostic: generation at batch 8 ran at 0.76 times its speed without it (not pinned to CPUs).

An engine that replays its steps from CUDA graphs gains little (`LIMITS.md`, "The bf16 cuBLAS
variants"): SGLang at its default settings computes in fp16 on this card, with no bf16 product, and
made to use bf16 it replays prefill and decode from graphs. Not measured: other models of each
kind, another card, engines other than those named here.
