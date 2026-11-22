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
| Kyoto | 227 | 64 | 1014.16 | 72.09 |
| Lyon | 227 | 64 | 1016.32 | 72.13 |
| Lima | 227 | 64 | 1014.26 | 72.17 |
| **Median** | | | **1014.26** | **72.13** |

The official vLLM 0.34.0 server used its compiler, Marlin experts and full and piecewise CUDA graphs on a Tesla V100-SXM2-32GB. The diagnostic run with eager execution and assembler optimization disabled measured 211.52 prefill and 9.23 decode tokens/s. [Graph run, responses and configuration](vllm-graphs.json) · [Diagnostic baseline](vllm-results.json) · [Checkpoint](https://huggingface.co/QuantTrio/Qwen3.5-35B-A3B-AWQ/tree/2f4fac011fc06e736782b0b59394577790da7d79)

## SGLang

Run the client against the official SGLang server started by the Ubuntu guide. With VF 0.1.1, a 32-token prefill chunk and batch-one CUDA decode graphs, it measured **180.82 prefill** and **38.12 decode** tokens/s. [Native JIT router, responses and phase timers](sglang-stock-jit32.json)

With eight-token chunks, a paired series using the same cache salts measured **62.55/38.25** with graphs and **72.97/12.27** with graphs disabled. The server processed 227 uncached tokens and returned 64 for each of the three requests. [Same-GPU measurements](sglang-stock-graphs.json) · [Eager diagnostic](sglang-stock-results.json)

```sh
mkdir -p ~/sglang-v100/benchmark && cd ~/sglang-v100/benchmark
base=https://raw.githubusercontent.com/nbald/volta-forever/main/docs/guides/benchmarks
curl -fsSLO "$base/sglang.py" && curl -fsSLO "$base/prompts.json"
python3 sglang.py
```

The client sets a fresh cache salt on every chat. It verifies the input and output token counts and the absence of device, host and storage cache hits. Prefill divides the input tokens by the GPU timer for `extend`. Decode uses the output tokens after the first token and the GPU timer for `decode`. [SGLang Ubuntu guide](../sglang.md) · [Historical adapter experiment](sglang-results.json)

## FreeToken

Run the streaming client and read the scheduler timings as described in the [FreeToken guide](../freetoken.md#5-benchmarks).
With batch-one CUDA graphs, FreeToken measured **884.56 prefill** and **109.26 decode** tokens/s. Native llama.cpp on the same V100 and Q4_K_M checkpoint measured **474.57** and **99.77** tokens/s. Each of the three measured requests read 227 uncached tokens and generated 64. [FreeToken measurements](freetoken-graphs.json) · [Native reference and commands](llamacpp-reference.json) · [Earlier eager run](freetoken-results.json)

## Strata

The CUDA 13 engine with VF and the native CUDA 12.9 engine served the same Qwen3.5-35B-A3B-MTP checkpoint on the same V100. Over 15 uncached requests per configuration, Strata 0.1.64 with VF reached 1871.4 prefill and 152.2 decode tokens/s. The native 0.1.64 engine reached 1877.6 and 150.9 on the same card. [Ubuntu guide](../strata.md#5-benchmarks) · [Results and native comparison](strata-results.json)

## TensorFold

The client reads TensorFold’s native prompt and generation timers and checks the token counts on each request. On the same V100, TensorFold with INT8 prompt activations reached 326.80 prefill and 17.01 decode tokens/s. The native llama.cpp reference reached 460.45 and 99.12. Both ran Qwen3.5-35B-A3B. TensorFold used MLX affine weights with 4-bit experts and 8-bit routers, and llama.cpp used GGUF Q4_K_M. [Ubuntu guide](../tensorfold.md#5-benchmarks) · [Results and checkpoint revision](tensorfold-results.json)
