# Single-request 35B benchmarks

Start the engine using its Ubuntu guide. Run the benchmark on the same computer, with one model loaded and the server idle. Each run uses the model's chat template at temperature zero, with thinking disabled and a limit of 64 output tokens. The warm-up starts with Lisbon. The three measured prompts start with Kyoto, Lyon and Lima and ask for a handover based on different notebook entries. Read the generated text when checking a new engine. [Common prompts](prompts.json)

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

The official SGLang server selected Marlin for the AWQ experts and used CUDA graphs in both phases. It processed each 227-token prompt in one prefill chunk and generated 64 tokens. Median GPU throughput was **2015.34 tokens/s prefill** and **77.36 tokens/s decode** on the same V100 where native 1Cat reached 1427.41 and 76.13. A separate same-GPU comparison of 128- and 256-token chunks measured 1565.10 and 2001.75 prefill tokens/s. Decode held at 77.38 and 77.39. [Marlin responses, timers and native references](sglang-stock-marlin.json)

An earlier WNA16 configuration with 32-token chunks measured 311.97 prefill and 52.68 decode tokens/s. [WNA16 results](sglang-stock-sm80.json)

With eight-token chunks, a paired series using the same cache salts measured **62.55/38.25** with graphs and **72.97/12.27** with graphs disabled. The server processed 227 uncached tokens and returned 64 for each of the three requests. [Same-GPU measurements](sglang-stock-graphs.json) · [Eager diagnostic](sglang-stock-results.json)

```sh
mkdir -p ~/sglang-v100/benchmark && cd ~/sglang-v100/benchmark
base=https://raw.githubusercontent.com/nbald/volta-forever/main/docs/guides/benchmarks
curl -fsSLO "$base/sglang.py" && curl -fsSLO "$base/prompts.json"
python3 sglang.py
```

The client gives each chat a fresh cache salt and checks the token counts and cache hits. It measures 227 input tokens with the GPU prefill timer and 63 subsequent tokens with the decode timer. With the wheel’s default router and normalization, SGLang reached 1988.55 prefill and 74.62 decode tokens/s. [Phase timings and replies](sglang-default-operators.json) · [Ubuntu guide](../sglang.md)

For NEXTN speculative decoding, the client sums the GPU timers for draft generation, verification and extension. [NEXTN measurements](sglang-stock-mtp.json) · [Earlier Marlin comparisons](sglang-stock-marlin.json) · [Historical adapter experiment](sglang-results.json)

## FreeToken

Run the streaming client and read the scheduler timings as described in the [FreeToken guide](../freetoken.md#5-benchmarks).
With batch-one CUDA graphs, FreeToken measured **884.56 prefill** and **109.26 decode** tokens/s. Native llama.cpp on the same V100 and Q4_K_M checkpoint measured **474.57** and **99.77** tokens/s. Each of the three measured requests read 227 uncached tokens and generated 64. [FreeToken measurements](freetoken-graphs.json) · [Native reference and commands](llamacpp-reference.json) · [Earlier eager run](freetoken-results.json)

## ZML

An experimental ZML build loaded the same QuantTrio AWQ checkpoint as vLLM and SGLang. The latest local build reads packed expert weights during prefill. The three measured requests each read 227 uncached tokens and generated 64. Median prefill rose from 448.27 to 535.35 tokens/s and median decode fell from 42.74 to 36.39. The three responses matched the preceding build token for token. Native 1Cat-vLLM reached 1427.41 and 76.13 on the same card. The upstream ZML example still needs a loader for these weights. [Experimental phase timings and answers](zml-awq-experimental.json) · [Ubuntu guide](../zml.md)

## Strata

Strata CUDA 13 with VF and cuBLAS FP32 prompt projections reached **1894.8 prefill and 152.5 decode tokens/s** on Qwen3.5-35B-A3B-MTP. The default projections reached 1869.9 and 152.4 on the same V100. Each setting ran 15 uncached chats, with 227 prompt and 64 generated tokens each. [Ubuntu guide](../strata.md#5-benchmarks) · [Per-request results and native CUDA comparison](strata-results.json)

## NInfer

The official NInfer server ran its published Qwen3.6-35B-A3B quantization with all weights on the V100. Each measured chat processed 227 uncached tokens and returned 64. Median throughput was 431.60 tokens/s prefill and 117.08 tokens/s decode. Native llama.cpp on the same card measured 470.60 and 99.49 with Qwen3.5 Q4_K_M. The comparison uses different checkpoints and quantizations. [Ubuntu guide and commands](../ninfer.md#5-benchmarks) · [Phase timings and responses](ninfer-results.json)

## TensorFold

The official 1.3.3 build reached 366.90 tokens/s prefill and 19.29 decode with VF `a9a4044`. The same executable with VF `1740d8f` reached 324.46 and 16.92 on the same V100. The two runs produced identical token IDs with MLX 4-bit experts and 8-bit routers. Native llama.cpp reached 470.60 and 99.49 on that GPU with GGUF Q4_K_M. [Ubuntu guide](../tensorfold.md#5-benchmarks) · [TensorFold phase timers and responses](tensorfold-stock-vf.json) · [Native reference](ninfer-results.json)
