# Single-request 35B benchmarks

The common comparison uses single-token autoregressive decoding with **MTP and all other speculative decoding disabled**. Start the engine using its Ubuntu guide. Run the benchmark on the same computer, with one model loaded and the server idle. Each run uses the model's chat template at temperature zero, with thinking disabled and a limit of 64 output tokens. The warm-up starts with Lisbon. The three measured prompts start with Kyoto, Lyon and Lima and ask for a handover based on different notebook entries. Read the generated text when checking a new engine. [Common prompts](prompts.json)

The measured vLLM server reported `speculative_config=None`, and SGLang reported `speculative_algorithm=None` and no draft-model path. FreeToken reported `speculative_algorithm=None` on the common UD-Q4_K_M file. Strata reported zero proposed and accepted drafts on every measured request. TensorFold reported `drafts=false`. NInfer’s Qwen3.5 and Qwen3.6 servers reported `speculative.backend=none` and zero draft rounds on every request.

## vLLM

In a second terminal, download the client and prompts together and run the client. It uses the Python standard library.

```sh
mkdir -p ~/vllm-v100/benchmark && cd ~/vllm-v100/benchmark
base=https://raw.githubusercontent.com/nbald/volta-forever/main/docs/guides/benchmarks
curl -fsSLO "$base/vllm.py" && curl -fsSLO "$base/prompts.json"
python3 vllm.py
```

Each request produces one new observation in each vLLM phase histogram. The client divides the prompt token count by the time in prefill. The first output token is available at the end of prefill, so decode throughput uses the remaining output tokens and the time in decode. It prints each response and the median of the three rates for each phase. [vLLM Ubuntu guide](../vllm.md)

| Prompt | Input tokens | Output tokens | Prefill (tokens/s) | Decode (tokens/s) |
| --- | ---: | ---: | ---: | ---: |
| Kyoto | 227 | 64 | 1494.19 | 80.23 |
| Lyon | 227 | 64 | 1449.98 | 80.30 |
| Lima | 227 | 64 | 1428.45 | 80.23 |
| **Median** | **227** | **64** | **1449.98** | **80.23** |

The official server used V100 expert tiles, Marlin, the default assembler settings and full and piecewise CUDA graphs. Native 1Cat-vLLM measured 1443.33 prefill and 76.09 decode tokens/s on the same GPU. [Per-request results](vllm-default-assembler.json) · [Native reference](vllm-graphs.json)

On another V100, a paired comparison with a 192-register cap measured 1468.21 prefill and 80.16 decode with V100 tiles and 991.50 and 72.18 with default tiles. [Tile comparison](vllm-expert-tiles.json)

## SGLang

An earlier SGLang run selected Marlin for the AWQ experts and used CUDA graphs in both phases. It processed each 227-token prompt in one prefill chunk and generated 64 tokens. Median GPU throughput was **2015.34 tokens/s prefill** and **77.36 tokens/s decode** on the same V100 where native 1Cat reached 1427.41 and 76.13. A separate same-GPU comparison of 128- and 256-token chunks measured 1565.10 and 2001.75 prefill tokens/s. Decode held at 77.38 and 77.39. [Marlin responses, timers and native references](sglang-stock-marlin.json)

An earlier WNA16 configuration with 32-token chunks measured 311.97 prefill and 52.68 decode tokens/s. [WNA16 results](sglang-stock-sm80.json)

With eight-token chunks, a paired series using the same cache salts measured **62.55/38.25** with graphs and **72.97/12.27** with graphs disabled. The server processed 227 uncached tokens and returned 64 for each of the three requests. [Same-GPU measurements](sglang-stock-graphs.json) · [Eager diagnostic](sglang-stock-results.json)

```sh
mkdir -p ~/sglang-v100/benchmark && cd ~/sglang-v100/benchmark
base=https://raw.githubusercontent.com/nbald/volta-forever/main/docs/guides/benchmarks
curl -fsSLO "$base/sglang.py" && curl -fsSLO "$base/prompts.json"
python3 sglang.py
```

The client gives each chat a fresh cache salt and checks the token counts and cache hits. It measures 227 input tokens with the GPU prefill timer and 63 subsequent tokens with the decode timer. The configuration in the Ubuntu guide uses the default expert router and FlashInfer’s CUDA normalization. With VF 0.1.3, its medians were **2010.01 prefill and 79.55 decode tokens/s**. With FlashInfer’s default CuTe normalization it reached 2000.45 and 76.69 in the same session. A second session, which started with empty caches, measured 1997.08 and 79.41. All three ran on the GPU of the earlier runs. With VF main at `1740d8f` the same two configurations measured 2016.06 / 77.33 and 1988.55 / 74.62. [Phase timings and replies](sglang-vf-0.1.3.json) · [Earlier timings](sglang-default-operators.json) · [Ubuntu guide](../sglang.md)

The common client checks for ordinary prefill and decode phases. The archived NEXTN experiment measured speculative decoding separately. [NEXTN measurements](sglang-stock-mtp.json) · [Earlier Marlin comparisons](sglang-stock-marlin.json) · [Historical adapter experiment](sglang-results.json)

## FreeToken

Run the streaming client and read the scheduler timings as described in the [FreeToken guide](../freetoken.md#5-benchmarks).
With batch-one CUDA graphs, FreeToken measured **905.68 prefill** and **108.75 decode** tokens/s. Native llama.cpp measured **494.57 / 101.27** using the same GGUF on the same card. [Requests and native reference](common-gguf-q4-k-m.json)

## ZML

On the same V100, the general BF16 multiply optimization took official LLMD from **746.45 to 774.63 GPU prefill tokens/s** and **48.55 to 50.16 post-first-token decode tokens/s** with tree accumulation. Both runs used the AMD MXFP4 checkpoint and three 227-token prompts with 64-token replies after warm-up. [Ubuntu guide and commands](../zml.md#5-benchmarks) · [Per-request measurements](zml-llmd-bf16-product.json) · [Earlier column modes](zml-llmd-column.json)

### LLMD GPU prefill

The ordinary client reports post-first-token decode tokens/s, internal decode steps/s and prompt tokens per second through the first token. To time the GPU prefill graph, start a fresh LLMD server with the CUPTI collector below. Use the same launch options as in the Ubuntu guide. The collector starts after warm-up and records the three measured requests. [LLMD launch command](../zml.md#5-benchmarks) · [NVIDIA CUPTI Activity API](https://docs.nvidia.com/cupti/12.9/main/main.html)

In the server terminal, build the collector from your VF checkout. The optional profiling dependencies are in NVIDIA’s Ubuntu CUDA repository. [Ubuntu 24.04 packages](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/)

```bash
bench="$HOME/volta-forever/docs/guides/benchmarks"
cuda=/usr/local/cuda-12.9/targets/x86_64-linux
if [ ! -f "$cuda/include/cupti.h" ] || [ ! -f "$cuda/include/cuda.h" ]; then
  sudo apt install -y cuda-cupti-dev-12-9 cuda-cudart-dev-12-9
fi
export VF_LLMD_PROFILE_DIR="$(mktemp -d)"
gcc -std=gnu11 -O2 -fPIC -shared "$bench/zml-activity.c" -I"$cuda/include" \
  -L"$cuda/lib" -Wl,-rpath,"$cuda/lib" -lcupti -pthread -o "$VF_LLMD_PROFILE_DIR/capture.so"
printf 'export VF_LLMD_PROFILE_DIR=%q\n' "$VF_LLMD_PROFILE_DIR"
export LD_PRELOAD="$VF_LLMD_PROFILE_DIR/capture.so${LD_PRELOAD:+:$LD_PRELOAD}"
```

Start LLMD in that terminal. In a second terminal, paste the printed `export VF_LLMD_PROFILE_DIR=…` line and run the client.

```sh
bench="$HOME/volta-forever/docs/guides/benchmarks"
python3 "$bench/zml.py" --profile-dir "$VF_LLMD_PROFILE_DIR" > "$VF_LLMD_PROFILE_DIR/responses.jsonl"
python3 "$bench/zml-profile.py" "$VF_LLMD_PROFILE_DIR/activity.tsv" "$VF_LLMD_PROFILE_DIR/responses.jsonl"
```

The result includes the input and output counts, the elapsed time from the first to the last kernel in the prefill graph, and the median GPU prefill rate. The analyzer checks complete graph signatures for the 40-layer MoE checkpoint and rejects captures with dropped records. A repeat on a second V100 measured **744.45 GPU prefill tokens/s** with `VF_BF16_TC_TILES=1` and the single-column optimization disabled.

## Strata

Strata CUDA 13 with VF reached **2000.00 prefill / 109.13 decode tokens/s**. Native CUDA 12 reached **1968.80 / 105.19** on the same card and UD-Q4_K_M file. Every request reported zero draft proposals and acceptances. The decode numerator is 63. [Ubuntu guide](../strata.md#5-benchmarks) · [Paired measurements](common-gguf-q4-k-m.json)

## NInfer

The official NInfer server processed each 227-token prompt without a prefix-cache hit and generated 64 tokens. With grouped BF16 Tensor Cores enabled, Qwen3.5 reached 556.51 tokens/s prefill and 117.08 decode. The scalar control reached 419.40 and 117.09. Native llama.cpp on the same V100 and prompts reached 470.60 and 99.49 with the Qwen3.5 Q4_K_M GGUF. NInfer used its own mixed Q4/Q5/Q6/Q8 conversion. Upstream’s published Qwen3.6 artifact measured 570.65 and 116.93 with groups enabled, or 431.70 and 117.04 with groups disabled. [Ubuntu guide](../ninfer.md#5-benchmarks) · [Qwen3.5 timings and replies](ninfer-qwen35-native.json) · [Qwen3.6 timings and replies](ninfer-bf16-mma.json)

## TensorFold

Official TensorFold 1.3.3 ran the MLX affine Q4 checkpoint at **462.85 prefill and 57.65 decode tokens/s**. The three measured requests used 227 uncached input tokens, 64 output tokens and drafting disabled. Native llama.cpp on the same V100 measured **470.60 / 99.49** with GGUF Q4_K_M. [Ubuntu guide](../tensorfold.md#5-benchmarks) · [TensorFold timings](tensorfold-generic.json) · [Native reference](tensorfold-stock-vf.json)

## llama.cpp

At upstream b12418, native CUDA 12.9.2 measured **494.57 prefill / 101.27 decode tokens/s**. The official CUDA 13.5 archive with VF 0.1.1 measured **1089.57 / 101.11**. Both used flash attention and the same Unsloth UD-Q4_K_M file as Strata and FreeToken. [Ubuntu guide](../llamacpp.md) · [Paired measurements](common-gguf-q4-k-m.json)
