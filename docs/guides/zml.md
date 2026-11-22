# ZML on a Tesla V100

## 1. CUDA

On Ubuntu 24.04 x86-64, run this check for the proprietary NVIDIA R580 driver.

```bash
if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | grep -E 'V100.*, 580[.]'; then
  sudo apt update && sudo apt install -y nvidia-driver-580
fi
```

If you installed the driver, reboot and repeat the check. It should print your V100 and driver version.

Bazel downloads the CUDA 13 libraries for ZML, so this build needs no separate system toolkit.

## 2. VF

```sh
sudo apt update && sudo apt install -y curl python3 xz-utils
curl -fsSL https://raw.githubusercontent.com/nbald/volta-forever/main/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
vf fetch --all
```

## 3. Engine

```sh
sudo apt install -y git unzip zstd
git clone --depth 1 https://github.com/zml/zml.git
cd zml
./bazel.sh build -c opt --@zml//platforms:cuda=true //examples/llm
```

## 4. Test

```sh
vf ./bazel-bin/examples/llm/llm --model=hf://Qwen/Qwen3.5-0.8B \
  --seqlen=128 --topk=1 --prompt="The capital of France is"
```

Look for `platform: cuda` and `Tesla V100` in the device listing. The tested model answered **Paris**.

## 5. Benchmarks

Read the final `decode` line after running the command above. On one Tesla V100-SXM2-32GB with driver 580.178.04 and VF 0.1.0, Qwen3.5-0.8B decoded at **96.1 tokens/s** over 0.458 seconds. This is a single run of the upstream example. [Tested ZML revision](https://github.com/zml/zml/commit/f430c3530fbc6d99f734b641eb8f3fd699868843)

The official **LLMD** server generated with the full AMD MXFP4 Qwen3.5-35B-A3B checkpoint on one V100. This test used the VF launcher and its external PTX assembler with CUDA 12.9.2. LLMD ran with the Triton backend, one request at a time, a 256-token batch, 256-token prefill chunks and MTP disabled. Setting `reasoning_effort` to `"none"` produced a visible **Paris** and the expected answer on all four benchmark prompts.

| Prompt | Input / output tokens | Input tokens/s through first token | Decode (tokens/s) |
| --- | ---: | ---: | ---: |
| Lisbon (warm-up) | 228 / 64 | 478.99 | 38.31 |
| Kyoto | 227 / 64 | 482.98 | 38.35 |
| Lyon | 227 / 64 | 484.01 | 38.32 |
| Lima | 227 / 64 | 482.98 | 38.38 |
| **Median after warm-up** | **227 / 64** | **482.98** | **38.35** |

The input rate includes all server work through the first token. LLMD exposes separate decode time and step counters. With the same VF compiler, raising both the token batch and prefill chunk from 64 to 256 raised prompt throughput from 309.69 to 482.98 tokens/s. With the batch at 256 and chunks at 64, the rate was 123.10. Native 1Cat-vLLM reached **1427.41 prefill / 76.13 decode** on the same GPU and prompts using QuantTrio AWQ. The checkpoint format and prefill timer differ. [LLMD settings, counters and replies](./benchmarks/zml-llmd-mxfp4.json) · [Native reference](./benchmarks/zml-awq-experimental.json)

## 6. Additional notes

With Qwen3.5-35B-A3B-AWQ, the unmodified ZML example selected the V100 and stopped before generation. The checkpoint stores expert weights separately, and the loader asks for `model.language_model.layers.0.mlp.experts.gate_up_proj`. [QuantTrio checkpoint](https://huggingface.co/QuantTrio/Qwen3.5-35B-A3B-AWQ) · [ZML's Qwen3.5 MoE loader](https://github.com/zml/zml/blob/f430c3530fbc6d99f734b641eb8f3fd699868843/examples/llm/models/qwen3_5_moe/model.zig)

The MLX 4-bit and Sehyo NVFP4 indexes also lack that tensor. The tested ZML loader creates the expert layers without attaching the scales needed to interpret packed weights. [ZML linear layer](https://github.com/zml/zml/blob/f430c3530fbc6d99f734b641eb8f3fd699868843/zml/nn/nn.zig#L23-L44) · [MLX weight index](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit/blob/main/model.safetensors.index.json) · [NVFP4 weight index](https://huggingface.co/Sehyo/Qwen3.5-35B-A3B-NVFP4/blob/main/model.safetensors.index.json)

ZML publishes the example used here as source. It also ships a separate inference server, **LLMD**, as a prebuilt CUDA image. In a V100 diagnostic with the October 23 LLMD build, XLA compiled GPU kernels through VF, then execution stopped at the bundled CUTLASS expert library. Its sm_120 binary contains only machine code for that architecture. Recompiling one upstream NVFP4 expert matrix multiplication with PTX retained exposed a second requirement. The kernel reads the cluster rank and uses tensor-map bulk copies. VF currently stops at the cluster rank on V100. This October build still requires support for those expert kernels. [LLMD distribution](https://zml.ai/llmd/) · [Upstream CUTLASS expert library](https://github.com/zml/flashinfer/releases/tag/cutlass-moe-v0.2.0)

A separate GPU profile with 64-token chunks showed four prefill graphs totaling 725.61 ms for one 227-token prompt. Quantized expert projections took 428.21 ms and dense projections took 236.33 ms of accumulated kernel time. The 63 complete decode graphs had a median duration of 24.59 ms. [GPU graph timings](./benchmarks/zml-llmd-mxfp4.json)

The September 24 LLMD distribution loads the split expert tensors in AMD’s MXFP4 checkpoint directly. The 35B test used its original server binary and VF’s runfiles integration for XLA’s external assembler. [VF compiler integration](https://github.com/nbald/volta-forever/commit/9013f262ee3866d999013736f6886429eb4803c4) The commands in sections 3 and 4 run the upstream source example with 0.8B. [AMD checkpoint revision](https://huggingface.co/amd/Qwen3.5-35B-A3B-MXFP4/tree/2e19c6576db91e5d5a93455415619262218bf8a1) · [VF support for XLA kernels](https://github.com/nbald/volta-forever/commit/c99663f7bf4e0d70ecd8e10f1078ff9c43cd9b4d)

An earlier local modification to the ZML example added a QuantTrio AWQ loader and grouped expert kernels. That build reached **536.38 prefill / 66.85 decode** on this card. [Experimental AWQ builds, measurements and profiles](./benchmarks/zml-awq-experimental.json)
