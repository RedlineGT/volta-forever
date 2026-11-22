# Run NInfer on a V100

Run the official NInfer server with Qwen3.5-35B-A3B on one 32 GB Tesla V100. Set `VF_CAPABILITY=12.0` when launching NInfer so its hardware check accepts the device.

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

Build the official server, then create a VF virtual environment for the Hugging Face CLI and NInfer’s CPU weight converter. [NInfer build instructions](https://github.com/Neroued/ninfer#build) · [Upstream conversion guide](https://github.com/Neroued/ninfer/blob/ebdc67e584bcd9ebe16ee63af9abeb2118bac6cd/docs/weight-conversion.md)

```sh
sudo apt install -y build-essential cmake ninja-build git pkg-config python3-venv \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libjpeg-dev \
  libhwloc-dev liburing-dev libcurl4-openssl-dev
git clone https://github.com/Neroued/ninfer.git ~/ninfer
cd ~/ninfer
cmake --preset rtx5090-release
cmake --build build/rtx5090-release --parallel 4 --target ninfer-serve
vf venv .venv
. .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install numpy huggingface_hub
```

Download the original Qwen3.5 weights and text resources, then convert them using NInfer’s built-in MoE recipe. The recipe name contains `qwen3_6` and the weights come from Qwen3.5. The BF16 download and converted file together occupy about 93 GB. [Source checkpoint](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/tree/2508b2437e382c605eb0fc52b83f5545960f6b4a)

```sh
cd ~/ninfer
hf download Qwen/Qwen3.5-35B-A3B --revision 2508b2437e382c605eb0fc52b83f5545960f6b4a --local-dir "$HOME/models/Qwen3.5-35B-A3B" \
  --include 'model.safetensors-*.safetensors' --include model.safetensors.index.json \
  --include config.json --include tokenizer.json --include tokenizer_config.json \
  --include chat_template.jinja --include generation_config.json
python -m tools.convert --model "$HOME/models/Qwen3.5-35B-A3B" \
  --recipe qwen3_6_35b_a3b --components text --device cpu \
  --cpu-threads 6 --source-cache-mib 1024 \
  --name qwen3.5-35b-a3b --out "$HOME/models/qwen3.5-35b-a3b.ninfer"
```

## 4. Test

Start the server in the NInfer directory. Drafting is off by default (`--spec none`). Prefix reuse is disabled so each benchmark prompt is measured in full.

```sh
cd ~/ninfer
VF_CAPABILITY=12.0 VF_BF16_TC_GROUPS=1 vf ./build/rtx5090-release/apps/ninfer-serve \
  "$HOME/models/qwen3.5-35b-a3b.ninfer" --host 127.0.0.1 --port 9086 \
  --max-context 512 --no-thinking --no-prefix-reuse --host-cache-capacity 0B
```

In a second terminal, ask for the capital of France. The measured server returned **Paris**.

```sh
curl -fsS http://127.0.0.1:9086/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-35b-a3b","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":12,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

## 5. Benchmarks

Keep the server running and execute the shared client in the second terminal.

```sh
python3 ~/volta-forever/docs/guides/benchmarks/strata-ninfer.py ninfer
```

Each measured chat processed 227 uncached tokens and generated 64. All three replies began with the expected city. The server reported zero draft rounds, proposed tokens and accepted tokens on every request. [Qwen3.5 requests, token IDs and phase timings](benchmarks/ninfer-qwen35-native.json)

| Qwen3.5-35B-A3B on the same V100 | Median prefill (tokens/s) | Median decode (tokens/s) |
|---|---:|---:|
| NInfer with the command above | **556.51** | **117.08** |
| NInfer with `VF_BF16_TC_GROUPS=0` | 419.40 | 117.09 |
| Native llama.cpp, Q4_K_M GGUF | 470.60 | 99.49 |

NInfer used its mixed Q4/Q5/Q6/Q8 conversion. The native reference used the same prompts with a Q4_K_M GGUF. Across the two NInfer runs, all 64 token IDs matched for Kyoto and Lyon. The first 58 matched for Lima. [Native reference and quantization details](benchmarks/ninfer-qwen35-native.json)

## 6. Additional notes

The tested Qwen3.5 artifact was 21,058,396,672 bytes. Its routed experts use Q4 gate/up and mixed Q5/Q6 down projections. [Artifact SHA256 and tensor formats](benchmarks/ninfer-qwen35-native.json)

For Qwen3.6 instead, upstream provides a ready-to-use `qwen3_6_35b_a3b.ninfer` file. With the same server settings it measured 570.65 tokens/s prefill and 116.93 decode, with zero draft rounds. [Upstream Qwen3.6 download](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) · [Qwen3.6 results](benchmarks/ninfer-bf16-mma.json)
