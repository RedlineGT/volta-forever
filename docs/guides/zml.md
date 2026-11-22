# ZML on a Tesla V100

## 1. CUDA

On Ubuntu 24.04 x86-64, run this check for the proprietary NVIDIA R580 driver.

```bash
if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | grep -E 'V100.*, 580[.]'; then
  sudo apt update && sudo apt install -y nvidia-driver-580
fi
```

If you installed the driver, reboot and repeat the check. It should print your V100 and driver version.

Bazel downloads the CUDA 13 libraries for the source example. The 35B LLMD server also needs CUDA 12.9.2 `ptxas`, included in the VF source setup below.

## 2. VF

For the 35B server, build VF with support for XLA’s external assembler. The source setup includes the CUDA libraries and the CUDA 12.9.2 assembler. [VF compiler support](https://github.com/nbald/volta-forever/commit/9013f262ee3866d999013736f6886429eb4803c4)

```bash
sudo apt update && sudo apt install -y build-essential ca-certificates curl git python3 xz-utils
if ! command -v rustup >/dev/null; then
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
. "$HOME/.cargo/env"
git clone --depth 1 https://github.com/nbald/volta-forever.git ~/volta-forever
(cd ~/volta-forever && scripts/build.sh && scripts/fetch-all.sh redist && scripts/build-forwarders.sh)
export PATH="$HOME/volta-forever/scripts:$PATH"
vf doctor
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

The official **LLMD** server also runs the full AMD MXFP4 Qwen3.5-35B-A3B checkpoint on one V100. The measured September distribution includes an `llmd` executable and its adjacent `llmd.runfiles` directory. The 35B run uses a VF build from the main branch with external assembler support. Sections 3 and 4 install and run the separate open-source ZML example. [LLMD distribution](https://zml.ai/llmd/) · [VF compiler support](https://github.com/nbald/volta-forever/commit/9013f262ee3866d999013736f6886429eb4803c4)

With the tested LLMD distribution installed, set `MODEL` to the local AMD checkpoint directory and run the following in the directory containing `llmd`. The defaults enable XLA autotuning. MTP is disabled. [Tested AMD checkpoint](https://huggingface.co/amd/Qwen3.5-35B-A3B-MXFP4/tree/2e19c6576db91e5d5a93455415619262218bf8a1)

```bash
VF_CAPABILITY=8.0 VF_BF16_TC_TILES=1 XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1 \
  vf ./llmd --model="${MODEL:?Set MODEL to the local AMD checkpoint directory}" --model-name=qwen \
  --backend=triton --batch-size=1 --prefill-chunk-size=256 --token-batch-size=256 \
  --page-chunk-size=16 --max-context-len=512 --gpu-memory-fraction=0.94 \
  --cache-memory-fraction=0.97 --topk=1 --listen=127.0.0.1:8000
```

Run the client from a second terminal while the server is idle. It streams four answers with `reasoning_effort="none"`, checks the expected city in each and reads the server counters before and after each request. Use a VF build containing the guarded BF16 tiles option. [VF change](https://github.com/nbald/volta-forever/commit/e2437776d38ed3dde9e8cc67fd0ffe68c7f55bbf)

```bash
base=https://raw.githubusercontent.com/nbald/volta-forever/main/docs/guides/benchmarks
curl -fsSLO "$base/zml.py" && curl -fsSLO "$base/prompts.json"
python3 zml.py
```

| Prompt | Input / output tokens | GPU prefill (tokens/s) | Server decode (steps/s) |
|---|---:|---:|---:|
| Kyoto | 227 / 64 | 745.82 | 46.01 |
| Lyon | 227 / 64 | 745.71 | 46.10 |
| Lima | 227 / 64 | 743.52 | 46.07 |
| **Median** | **227 / 64** | **745.71** | **46.07** |

On the same V100, enabling guarded BF16 tiles raised GPU prefill from **681.68 to 745.71 tokens/s**. Decode measured **46.15** with tiles off and **46.07** with tiles on. CUPTI timed all kernels in each prefill graph in a separate run. The server measured **673.59 / 739.41 prompt tokens/s through the first token** with tiles off / on. An independent run of the client above reproduced **659.88 / 718.35** through the first token and **45.81 / 44.97** in decode. Each run produced the expected city and handover. [Request timings and replies](./benchmarks/zml-llmd-autotune.json)

With autotuning disabled, LLMD measured **482.98 / 38.35**. The input rate includes all server work through the first token. Native 1Cat-vLLM reached **1427.41 prefill / 76.13 decode** on the same GPU and prompts with QuantTrio AWQ. The checkpoint format and prompt timer differ. [Prior chunk size measurements](./benchmarks/zml-llmd-mxfp4.json) · [Native reference](./benchmarks/zml-awq-experimental.json)

## 6. Additional notes

With Qwen3.5-35B-A3B-AWQ, the unmodified ZML example selected the V100 and stopped before generation. The checkpoint stores expert weights separately, and the loader asks for `model.language_model.layers.0.mlp.experts.gate_up_proj`. [QuantTrio checkpoint](https://huggingface.co/QuantTrio/Qwen3.5-35B-A3B-AWQ) · [ZML's Qwen3.5 MoE loader](https://github.com/zml/zml/blob/f430c3530fbc6d99f734b641eb8f3fd699868843/examples/llm/models/qwen3_5_moe/model.zig)

The MLX 4-bit and Sehyo NVFP4 indexes also lack that tensor. The tested ZML loader creates the expert layers without attaching the scales needed to interpret packed weights. [ZML linear layer](https://github.com/zml/zml/blob/f430c3530fbc6d99f734b641eb8f3fd699868843/zml/nn/nn.zig#L23-L44) · [MLX weight index](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit/blob/main/model.safetensors.index.json) · [NVFP4 weight index](https://huggingface.co/Sehyo/Qwen3.5-35B-A3B-NVFP4/blob/main/model.safetensors.index.json)

ZML publishes the example used here as source. It also ships a separate inference server, **LLMD**, as a prebuilt CUDA image. In a V100 diagnostic with the October 23 LLMD build, XLA compiled GPU kernels through VF, then execution stopped at the bundled CUTLASS expert library. Its sm_120 binary contains only machine code for that architecture. Recompiling one upstream NVFP4 expert matrix multiplication with PTX retained exposed a second requirement. The kernel reads the cluster rank and uses tensor-map bulk copies. VF currently stops at the cluster rank on V100. This October build still requires support for those expert kernels. [LLMD distribution](https://zml.ai/llmd/) · [Upstream CUTLASS expert library](https://github.com/zml/flashinfer/releases/tag/cutlass-moe-v0.2.0)

A separate GPU profile with 64-token chunks showed four prefill graphs totaling 725.61 ms for one 227-token prompt. Quantized expert projections took 428.21 ms and dense projections took 236.33 ms of accumulated kernel time. The 63 complete decode graphs had a median duration of 24.59 ms. [GPU graph timings](./benchmarks/zml-llmd-mxfp4.json)

The September 24 LLMD distribution loads the split expert tensors in AMD’s MXFP4 checkpoint directly. The 35B test used its original server binary and VF’s runfiles integration for XLA’s external assembler. [VF compiler integration](https://github.com/nbald/volta-forever/commit/9013f262ee3866d999013736f6886429eb4803c4) The commands in sections 3 and 4 run the upstream source example with 0.8B. [AMD checkpoint revision](https://huggingface.co/amd/Qwen3.5-35B-A3B-MXFP4/tree/2e19c6576db91e5d5a93455415619262218bf8a1) · [VF support for XLA kernels](https://github.com/nbald/volta-forever/commit/c99663f7bf4e0d70ecd8e10f1078ff9c43cd9b4d)

An earlier local modification to the ZML example added a QuantTrio AWQ loader and grouped expert kernels. That build reached **536.38 prefill / 66.85 decode** on this card. [Experimental AWQ builds, measurements and profiles](./benchmarks/zml-awq-experimental.json)
