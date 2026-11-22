# Single-request 35B benchmarks

The common comparison uses single-token autoregressive decoding with **MTP and all other speculative decoding disabled**. Start the engine using its Ubuntu guide. Run the benchmark on the same computer, with one model loaded and the server idle. Each run uses the model's chat template at temperature zero, with thinking disabled and a limit of 64 output tokens. The warm-up starts with Lisbon. The three measured prompts start with Kyoto, Lyon and Lima and ask for a handover based on different notebook entries. Read the generated text when checking a new engine. [Common prompts](prompts.json)

The measured vLLM server reported `speculative_config=None`, and SGLang reported `speculative_algorithm=None` and no draft-model path. FreeToken used the standard Q4_K_M file with no drafter supplied. Strata reported zero proposed and accepted drafts on every measured request. TensorFold reported `drafts=false`. NInfer’s Qwen3.5 and Qwen3.6 servers reported `speculative.backend=none` and zero draft rounds on every request.

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

The client gives each chat a fresh cache salt and checks the token counts and cache hits. It measures 227 input tokens with the GPU prefill timer and 63 subsequent tokens with the decode timer. With its default expert router and FlashInfer’s CUDA normalization, the official wheel reached 2016.06 prefill and 77.33 decode tokens/s. With FlashInfer’s default CuTe normalization it reached 1988.55 and 74.62. [Phase timings and replies](sglang-default-operators.json) · [Ubuntu guide](../sglang.md)

The common client checks for ordinary prefill and decode phases. The archived NEXTN experiment measured speculative decoding separately. [NEXTN measurements](sglang-stock-mtp.json) · [Earlier Marlin comparisons](sglang-stock-marlin.json) · [Historical adapter experiment](sglang-results.json)

## FreeToken

Run the streaming client and read the scheduler timings as described in the [FreeToken guide](../freetoken.md#5-benchmarks).
With batch-one CUDA graphs, FreeToken measured **884.56 prefill** and **109.26 decode** tokens/s. Native llama.cpp on the same V100 and Q4_K_M checkpoint measured **474.57** and **99.77** tokens/s. Each of the three measured requests read 227 uncached tokens and generated 64. [FreeToken measurements](freetoken-graphs.json) · [Native reference and commands](llamacpp-reference.json) · [Earlier eager run](freetoken-results.json)

## ZML

An experimental ZML build loaded the same QuantTrio AWQ checkpoint as vLLM and SGLang. The latest local build reads packed expert weights during prefill and groups expert projections during decode. The three measured requests each read 227 uncached tokens and generated 64. The grouped decode kernel increased median decode from 36.39 to 66.85 tokens/s. Median prefill measured 536.38 tokens/s and all 192 output token IDs matched the preceding build. Native 1Cat-vLLM reached 1427.41 and 76.13 on the same card. The upstream ZML example still needs a loader for these weights. [Experimental phase timings and answers](zml-awq-experimental.json) · [Ubuntu guide](../zml.md)

## Strata

With **MTP and all other drafting disabled**, Strata CUDA 13 with VF reached **1943.5 prefill and 104.55 decode tokens/s**. Native Strata CUDA 12 reached 1950.2 and 99.04 on the same V100, with the same checkpoint and cuBLAS setting. Each measured request read 227 uncached tokens and generated 64, with zero draft proposals and acceptances. The common decode numerator is 63. [Ubuntu guide](../strata.md#5-benchmarks) · [Non-speculative measurements](strata-no-drafts.json) · [Historical MTP experiment](strata-results.json)

## NInfer

The official NInfer server processed each 227-token prompt without a prefix-cache hit and generated 64 tokens. With grouped BF16 Tensor Cores enabled, Qwen3.5 reached 556.51 tokens/s prefill and 117.08 decode. The scalar control reached 419.40 and 117.09. Native llama.cpp on the same V100 and prompts reached 470.60 and 99.49 with the Qwen3.5 Q4_K_M GGUF. NInfer used its own mixed Q4/Q5/Q6/Q8 conversion. Upstream’s published Qwen3.6 artifact measured 570.65 and 116.93 with groups enabled, or 431.70 and 117.04 with groups disabled. [Ubuntu guide](../ninfer.md#5-benchmarks) · [Qwen3.5 timings and replies](ninfer-qwen35-native.json) · [Qwen3.6 timings and replies](ninfer-bf16-mma.json)

## TensorFold

The official TensorFold 1.3.3 source build answered Paris with all three settings. On Qwen3.5 MLX Q4, three requests of 227 input and 64 output tokens measured 461.27/34.01 tokens/s prefill/decode with `VF_BF16_ROW=off`, 462.39/35.67 with `ordered`, and 461.64/37.19 with `tree`. All three used grouped BF16 Tensor Cores and disabled drafting. Both options changed the Lisbon warm-up wording. The three measured sequences matched the control with `tree`. Native llama.cpp reached 470.60/99.49 on the same V100 with GGUF Q4_K_M. [Ubuntu guide](../tensorfold.md#5-benchmarks) · [Timings, token hashes and responses](tensorfold-bf16-row.json) · [Native reference](ninfer-results.json)
