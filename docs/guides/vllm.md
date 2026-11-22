# vLLM on a Tesla V100

## 1. CUDA

On Ubuntu 24.04 x86-64, run this check for the proprietary NVIDIA R580 driver.

```bash
if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | grep -E 'V100.*, 580[.]'; then
  sudo apt update && sudo apt install -y nvidia-driver-580
fi
```

If you installed the driver, reboot and repeat the check. It should print your V100 and driver version.

The 35B model compiles FlashInfer kernels on first launch. Use a complete CUDA 13 toolkit so the compiler and headers come from the same installation. The check verifies the CUDA 13 compiler and headers before installing the toolkit. [NVIDIA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#ubuntu)

```bash
export PATH="$PATH:${CUDA_HOME:-/usr/local/cuda}/bin"
if ! nvcc --version 2>/dev/null | grep -q 'release 13[.]' ||
   ! nvcc -E -x cu --pre-include cublas_v2.h /dev/null >/dev/null 2>&1; then
  sudo apt update && sudo apt install -y ca-certificates curl
  repo=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64
  curl -fsSL "$repo/cuda-archive-keyring.gpg" | sudo tee /usr/share/keyrings/cuda-archive-keyring.gpg >/dev/null
  echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] $repo/ /" | sudo tee /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list
  sudo apt update && sudo apt install -y cuda-toolkit-13
  export PATH="/usr/local/cuda-13/bin:$PATH"
fi
nvcc_path=$(readlink -f "$(command -v nvcc)")
export CUDA_HOME="${nvcc_path%/bin/nvcc}"
```

## 2. VF

Use VF from the main branch for this 35B configuration. The released VF 0.1.1 stops when vLLM compiles its unified-attention kernel. Main includes the required GPU instruction translation. [VF correction](https://github.com/nbald/volta-forever/commit/cc36e4e)

```sh
sudo apt update && sudo apt install -y build-essential ca-certificates curl git python3-venv xz-utils
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
sudo apt install -y build-essential git libgl1 libglib2.0-0t64 libibverbs1 libnuma-dev \
  librdmacm1t64 ninja-build python3-dev
mkdir -p ~/vllm-v100 && cd ~/vllm-v100
vf venv .venv
source .venv/bin/activate
pip install vllm
```

## 4. Test

Use the full CUDA 13 toolkit from step 1. Start the server in the active environment. vLLM downloads the 4-bit AWQ checkpoint on first launch. This configuration serves one text request at a time on a 32 GiB V100, with Marlin experts and CUDA graphs. The optimized run used VF commit `3a079c8` with vLLM 0.34.0. [Tested VF revision](https://github.com/nbald/volta-forever/commit/3a079c87b22a0793dc8aabfa62a5ca3ff11d91eb) [Qwen3.5-35B-A3B-AWQ](https://huggingface.co/QuantTrio/Qwen3.5-35B-A3B-AWQ)

```sh
export PATH="$CUDA_HOME/bin:$PATH"
export PTXAS_OPTIONS='--override-directive-values --maxrregcount=192 --opt-level=3'
vllm serve QuantTrio/Qwen3.5-35B-A3B-AWQ --served-model-name qwen \
  --dtype float16 --language-model-only --safetensors-load-strategy lazy -cc.mode=3 \
  --max-model-len 512 --max-num-seqs 1 --gpu-memory-utilization 0.9 \
  --host 127.0.0.1 --port 8000
```

Once the server is ready, send a request from a second terminal. The tested server returned **Paris** in two output tokens.

```sh
curl -fsS http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"What is the capital of France? Answer with just the city."}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

## 5. Benchmarks

With the server running, use the [benchmark client and prompts](./benchmarks/). Run one warm-up and three different chats in sequence, at temperature zero with thinking disabled. Each measured chat used 227 input tokens and generated 64 tokens. vLLM reported a 0% prefix-cache hit rate.

| Phase | Three runs (tokens/s) | Median (tokens/s) |
| --- | --- | ---: |
| Prefill | 1014.16, 1016.32, 1014.26 | **1014.26** |
| Decode | 72.09, 72.13, 72.17 | **72.13** |

Each rate uses the change in vLLM’s own phase timer for one request. Prefill counts all 227 input tokens. Decode counts the 63 output tokens after the first token. The test used a Tesla V100-SXM2-32GB, driver 580.178.04, CUDA toolkit 13.5, vLLM 0.34.0 and PyTorch 2.14.0+cu130. The server captured both full and piecewise CUDA graphs. [Measurements and generated text](./benchmarks/vllm-graphs.json)

On the same card and the same three requests, full CUDA graphs with the compiler disabled reached 1016.51 prefill and 72.24 decode tokens/s. Eager mode at the same assembler optimization level reached 890.74 and 9.43 tokens/s. [Same-GPU comparison](./benchmarks/vllm-graphs.json)

The native 1Cat-vLLM reference on the same card and AWQ checkpoint reached **1443.33 prefill** and **76.09 decode** tokens/s. Official vLLM, using the locally built VF 0.1.2 archive, reached **1006.90** and **72.14** in that comparison. Both used the same three requests with speculative decoding disabled. These native rates are the targets for this workload. [Paired measurements and responses](./benchmarks/vllm-graphs.json)

## 6. Additional notes

VF 0.1.1 reports the capability 7.5 required by vLLM’s NVML check. The 35B run used all nine checkpoint shards, automatic quantization selection and CUDA 13.5 headers for FlashInfer RMSNorm. VF 0.1.0 reported 7.0 to this check. [VF 0.1.1 release](https://github.com/nbald/volta-forever/releases/tag/v0.1.1)

For a smaller model, Qwen2.5-0.5B-Instruct with float16 weights also answered **Paris**. Three warm 32-token completions had a median end-to-end throughput of 33.4 tokens/s.
