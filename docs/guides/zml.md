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

The upstream build from step 3 stops before the first token with the QuantTrio 35B checkpoint. The following measurements required changes to ZML itself. Five local builds added support for the split AWQ expert weights, then optimized the GPU code for those experts. All used the original QuantTrio checkpoint. Each answered **Paris** and generated 64 tokens on each of three independent 227-token prompts. The medians below use the same V100 and checkpoint.

| Engine | Prefill (tokens/s) | Decode (tokens/s) |
| --- | ---: | ---: |
| Local ZML with split AWQ experts | 343.79 | 30.17 |
| Local ZML with fused AWQ decode projections | 350.57 | 42.73 |
| Local ZML with fused decode and eight-step prefill recurrence | 448.27 | 42.74 |
| Local ZML with packed AWQ prefill projections added | 535.35 | 36.39 |
| Local ZML with grouped AWQ decode projections added | 536.38 | 66.85 |
| Native 1Cat-vLLM | 1427.41 | 76.13 |

Packed prefill removed 1.5 GiB of intermediate BF16 weights per compiled layer. The grouped decode kernel matched 14,304 of 14,336 CPU-oracle values exactly on seven layer inputs and met the numerical tolerance on all of them. [Per-request text, times, numerical tolerances and GPU profiles](./benchmarks/zml-awq-experimental.json)

## 6. Additional notes

With Qwen3.5-35B-A3B-AWQ, the unmodified ZML example selected the V100 and stopped before generation. The checkpoint stores expert weights separately, and the loader asks for `model.language_model.layers.0.mlp.experts.gate_up_proj`. [QuantTrio checkpoint](https://huggingface.co/QuantTrio/Qwen3.5-35B-A3B-AWQ) · [ZML's Qwen3.5 MoE loader](https://github.com/zml/zml/blob/f430c3530fbc6d99f734b641eb8f3fd699868843/examples/llm/models/qwen3_5_moe/model.zig)

The MLX 4-bit and Sehyo NVFP4 indexes also lack that tensor. The tested ZML loader creates the expert layers without attaching the scales needed to interpret packed weights. [ZML linear layer](https://github.com/zml/zml/blob/f430c3530fbc6d99f734b641eb8f3fd699868843/zml/nn/nn.zig#L23-L44) · [MLX weight index](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit/blob/main/model.safetensors.index.json) · [NVFP4 weight index](https://huggingface.co/Sehyo/Qwen3.5-35B-A3B-NVFP4/blob/main/model.safetensors.index.json)

ZML publishes the example used here as source. It also ships a separate inference server, **LLMD**, as a prebuilt CUDA image. In a V100 diagnostic with the October 23 LLMD build, XLA compiled GPU kernels through VF, then execution stopped at the bundled CUTLASS expert library. Its sm_120 binary contains only machine code for that architecture. Recompiling one upstream NVFP4 expert matrix multiplication with PTX retained exposed a second requirement. The kernel reads the cluster rank and uses tensor-map bulk copies. VF currently stops at the cluster rank on V100. This October build still requires support for those expert kernels. [LLMD distribution](https://zml.ai/llmd/) · [Upstream CUTLASS expert library](https://github.com/zml/flashinfer/releases/tag/cutlass-moe-v0.2.0)

The September 24 LLMD build recognizes the split expert tensors in AMD’s MXFP4 format. A small diagnostic model exercised five XLA kernels from this build on a V100. All five matched CPU results with the VF update linked below. Generation with the full AMD 35B checkpoint remains unverified. [AMD checkpoint and revision](https://huggingface.co/amd/Qwen3.5-35B-A3B-MXFP4/tree/2e19c6576db91e5d5a93455415619262218bf8a1) · [VF support for the XLA kernels](https://github.com/nbald/volta-forever/commit/c99663f7bf4e0d70ecd8e10f1078ff9c43cd9b4d)
