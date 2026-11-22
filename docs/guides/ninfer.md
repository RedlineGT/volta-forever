# Run NInfer on a V100

The official NInfer CLI ran Qwen3.6-35B-A3B on one 32 GB Tesla V100. Set `VF_CAPABILITY=12.0` when launching NInfer so its hardware check accepts the device.

## 1. CUDA

On Ubuntu 24.04 x86-64, run this check for the proprietary NVIDIA R580 driver.

```bash
if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | grep -E 'V100.*, 580[.]'; then
  sudo apt update && sudo apt install -y nvidia-driver-580
fi
```

If you installed the driver, reboot and repeat the check. It should print your V100 and driver version.

The official x86-64 build needs a toolkit that can compile for `sm_120a`. This check reuses a suitable compiler and otherwise installs NVIDIA’s current CUDA Toolkit. [NVIDIA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#ubuntu)

```bash
export PATH="$PATH:${CUDA_HOME:-/usr/local/cuda}/bin"
if ! printf '%s\n' '__global__ void probe() {}' | nvcc -arch=sm_120a -cubin -x cu -o /dev/null - 2>/dev/null; then
  sudo apt update && sudo apt install -y ca-certificates curl
  repo=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64
  curl -fsSL "$repo/cuda-archive-keyring.gpg" | sudo tee /usr/share/keyrings/cuda-archive-keyring.gpg >/dev/null
  echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] $repo/ /" | sudo tee /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list
  sudo apt update && sudo apt install -y cuda-toolkit
  export PATH="/usr/local/cuda/bin:$PATH"
fi
```

## 2. VF

Build VF from the main branch for long prompts in the official NInfer server. [Required VF change](https://github.com/nbald/volta-forever/commit/a9dd4cece3cfa38d051e23ea6f37ef4f535f745a)

```sh
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

Upstream distributes the CLI as source. Build its official x86-64 preset. [NInfer build instructions](https://github.com/Neroued/ninfer#build)

```sh
sudo apt install -y build-essential cmake ninja-build git pkg-config \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libjpeg-dev \
  libhwloc-dev liburing-dev libcurl4-openssl-dev
git clone https://github.com/Neroued/ninfer.git
cd ninfer
cmake --preset rtx5090-release
cmake --build build/rtx5090-release --parallel 4 --target ninfer
```

## 4. Test

Use upstream's 22.8 GB Qwen3.6-35B-A3B artifact. Routed experts use Q4 gate/up and Q5/Q6 down projections. [Model card](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer)

```sh
curl -fL --retry 3 -o qwen3_6_35b_a3b.ninfer \
  https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/main/qwen3_6_35b_a3b.ninfer
VF_CAPABILITY=12.0 vf ./build/rtx5090-release/apps/ninfer qwen3_6_35b_a3b.ninfer \
  --prompt 'What is the capital of France? Answer with one word.' \
  --max-context 512 --max-tokens 12 --no-thinking --temperature 0
```

The test returned **Paris** on a Tesla V100-SXM2-32GB with driver 580.178.04, VF 0.1.1 and the upstream CLI built with CUDA 13.5. The CLI reported 24 input tokens and two output tokens for this answer.

## 5. Benchmarks

Start the official server from the same NInfer directory. Disable prefix reuse to measure each full prompt, and set the host cache capacity to zero for this single-client run.

```sh
cmake --build build/rtx5090-release --parallel 4 --target ninfer-serve
VF_CAPABILITY=12.0 VF_BF16_TC_GROUPS=1 vf ./build/rtx5090-release/apps/ninfer-serve qwen3_6_35b_a3b.ninfer \
  --host 127.0.0.1 --port 9086 --max-context 512 --no-thinking --no-prefix-reuse --host-cache-capacity 0B
```

In a second terminal, run the shared warm-up and three measured chats. The client records the server’s prefill and decode times for each reply.

```sh
python3 ~/volta-forever/docs/guides/benchmarks/strata-ninfer.py ninfer
```

After the warm-up, three prompts of 227 tokens produced 64 tokens each, with zero cached tokens. All replies began with the expected city. Both engines ran on the same Tesla V100-SXM2-32GB. [Original per-request timings and replies](benchmarks/ninfer-results.json)

| Engine and checkpoint | Median prefill (tokens/s) | Median decode (tokens/s) |
|---|---:|---:|
| NInfer with the command above, Qwen3.6 mixed Q4/Q5/Q6/Q8 | **570.65** | **116.93** |
| Same server with the default VF translation | 431.70 | 117.04 |
| Same NInfer weights, 128-token prefill chunks | 308.2 | 117.1 |
| Native llama.cpp, Qwen3.5 Q4_K_M | 470.6 | 99.5 |

To reproduce the second row, set `VF_BF16_TC_GROUPS=0` in the server command. Enabling the option raised prefill throughput by 32.2%. FP32 accumulation order and the sign of a zero result can change. At temperature zero, the Kyoto and Lima continuations changed. All 64 token IDs matched for Lyon. [Matched timings, token IDs and GPU profile](benchmarks/ninfer-bf16-mma.json)

The 128-token run used `--prefill-chunk 128` and processed each prompt in two chunks (128 and 99 tokens). The native llama.cpp reference uses a different checkpoint and quantization in GGUF. [NInfer chunk settings](https://github.com/Neroued/ninfer/blob/master/docs/startup-configuration.md#model-and-execution)

## 6. Additional notes

This demo needs the linked-kernel fixes included in VF 0.1.1. VF 0.1.0 stopped with `cudaErrorSymbolNotFound` in NInfer’s token-count test even with the 12.0 profile. [VF release](https://github.com/nbald/volta-forever/releases/tag/v0.1.1)

The Qwen3.6 file uses mixed precision. Across the 40 text layers, routed gate and up projections use Q4. Routed down projections use Q5 in 37 layers and Q6 in three. NInfer requires Q5 or Q6 down projections for this Q4 profile, and Q8 for the shared experts. The token embedding uses Q8 and the output head uses Q6. [Upstream weight checks](https://github.com/Neroued/ninfer/blob/ebdc67e584bcd9ebe16ee63af9abeb2118bac6cd/src/ops/moe/operands.cpp#L81-L103)

Qwen3.5-35B-A3B Q4_K_M GGUF was rejected with `NInfer accepts only .ninfer artifacts`. The official `qwen3_6_35b_a3b` conversion recipe accepted the Qwen3.5 BF16 tensor names and shapes across all 14 source shards. Inference was verified with the published Qwen3.6 artifact. [NInfer conversion recipes](https://github.com/Neroued/ninfer/blob/master/docs/weight-conversion.md)
