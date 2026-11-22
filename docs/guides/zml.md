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

A local ZML build with native support for split AWQ expert weights ran Qwen3.5-35B-A3B on the V100. It returned **Paris**, then produced 64 tokens for each of the three 227-token prompts. Median throughput was **343.79 tokens/s in prefill** and **30.17 tokens/s in decode**. Native 1Cat-vLLM reached 1427.41 and 76.13 with the same checkpoint and GPU. This 35B loader is available in our local ZML build. [Measurements and source revision](./benchmarks/zml-awq-experimental.json)

## 6. Additional notes

With Qwen3.5-35B-A3B-AWQ, the unmodified ZML example selected the V100 and stopped before generation. The checkpoint stores expert weights separately, and the loader asks for `model.language_model.layers.0.mlp.experts.gate_up_proj`. [QuantTrio checkpoint](https://huggingface.co/QuantTrio/Qwen3.5-35B-A3B-AWQ) · [ZML's Qwen3.5 MoE loader](https://github.com/zml/zml/blob/f430c3530fbc6d99f734b641eb8f3fd699868843/examples/llm/models/qwen3_5_moe/model.zig)

The MLX 4-bit and Sehyo NVFP4 indexes also lack that tensor. The tested ZML loader creates the expert layers without attaching the scales needed to interpret packed weights. [ZML linear layer](https://github.com/zml/zml/blob/f430c3530fbc6d99f734b641eb8f3fd699868843/zml/nn/nn.zig#L23-L44) · [MLX weight index](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit/blob/main/model.safetensors.index.json) · [NVFP4 weight index](https://huggingface.co/Sehyo/Qwen3.5-35B-A3B-NVFP4/blob/main/model.safetensors.index.json)

ZML publishes the example used here as source. It also ships a separate inference server, **LLMD**, as a prebuilt CUDA image. In a V100 diagnostic with the October 23 LLMD build, XLA compiled GPU kernels through VF, then execution stopped at the bundled CUTLASS expert library. Its sm_120 binary contains only machine code for that architecture. Recompiling one upstream NVFP4 expert matrix multiplication with PTX retained exposed a second requirement. The kernel reads the cluster rank and uses tensor-map bulk copies. VF currently stops at the cluster rank on V100. The LLMD server still requires support for these expert kernels before it can run the 35B benchmark. [LLMD distribution](https://zml.ai/llmd/) · [Upstream CUTLASS expert library](https://github.com/zml/flashinfer/releases/tag/cutlass-moe-v0.2.0)
