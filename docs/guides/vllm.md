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

VF 0.1.2 includes the GPU instruction translation needed by vLLM’s unified-attention kernel. [VF correction](https://github.com/nbald/volta-forever/commit/cc36e4e)

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

Start the official vLLM server in the active environment. It downloads the 4-bit checkpoint on first launch. The V100 profile selects tile sizes for the first layer’s FP16 experts. vLLM uses Marlin for the quantized experts in the remaining layers. [Checkpoint and revision](https://huggingface.co/QuantTrio/Qwen3.5-35B-A3B-AWQ/tree/2f4fac011fc06e736782b0b59394577790da7d79)

```sh
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_TUNED_CONFIG_FOLDER="$HOME/volta-forever/tuning/vllm"
vllm serve QuantTrio/Qwen3.5-35B-A3B-AWQ --served-model-name qwen \
  --revision 2f4fac011fc06e736782b0b59394577790da7d79 \
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

With the server running, send one warm-up and three different chats from a second terminal.

```sh
cd ~/volta-forever
python3 docs/guides/benchmarks/vllm.py
```

Each measured request processed 227 uncached prompt tokens and generated 64 output tokens. The client reads vLLM’s phase timers. Decode counts the 63 tokens after the first. The official server used Marlin, the V100 expert tiles and both full and piecewise CUDA graphs. [Benchmark client](./benchmarks/vllm.py)

| Server on the same Tesla V100 | Median prefill (tokens/s) | Median decode (tokens/s) |
| --- | ---: | ---: |
| Official vLLM with the command above | **1449.98** | **80.23** |
| Native 1Cat-vLLM reference | 1443.33 | 76.09 |

Both runs loaded the same nine QuantTrio AWQ W4A16 shards (group size 128, zero points) and processed the same prompts. Use the `--revision` from section 4 for both servers to reproduce the comparison. The guide run used vLLM 0.34.0, PyTorch 2.14.0+cu130, VF’s retained-PTX fix and the default assembler settings. [Commands, timings and responses](./benchmarks/vllm-default-assembler.json) · [Native reference](./benchmarks/vllm-graphs.json)

An earlier comparison isolated the expert tile settings on another V100. With the assembler capped at 192 registers in both runs, the V100 tiles reached 1468.21 prefill and 80.16 decode, compared with 991.50 and 72.18 for vLLM’s default tiles. [Tile comparison](./benchmarks/vllm-expert-tiles.json)

## 6. Additional notes

VF 0.1.1 reports the capability 7.5 required by vLLM’s NVML check. The 35B run used all nine checkpoint shards, automatic quantization selection and CUDA 13.5 headers for FlashInfer RMSNorm. VF 0.1.0 reported 7.0 to this check. [VF 0.1.1 release](https://github.com/nbald/volta-forever/releases/tag/v0.1.1)

For a smaller model, Qwen2.5-0.5B-Instruct with float16 weights also answered **Paris**. Three warm 32-token completions had a median end-to-end throughput of 33.4 tokens/s.

### A model kept in bfloat16

Below capability 8.0 vLLM makes a bfloat16 model float16; at the default capability that is what runs. `VF_CAPABILITY=8.0 VF_CAPABILITY_ALL=1` keeps the model in bfloat16. Use it for a model that refuses float16. For an ordinary dense model the default capability is much faster. Qwen2.5-0.5B-Instruct, stored in bfloat16, on stock vLLM 0.34.0 with compilation and CUDA graphs on, 200 tokens a request, one V100: **359** tokens/s at one request and **7,445** at 32 in total at the default capability; **81.5** and **742** told 8.0, 4.4 and 10 times slower. Told 8.0, the model's bfloat16 products run in FlashAttention-2 and in cuBLAS's bfloat16 kernels without tensor cores: a V100 has them for float16 only. The text began the same in every run. Only this small model was measured. [Every run and its conditions](../performance.md#vllm)

`VF_TRITON_TC=1` acts at the default capability only; told 8.0, it says so in one line and does nothing. At the default capability, on the same model, it gave 356 to 365 tokens/s at one request and 7,901 at 32, 6 % more. It does not help a bfloat16 model that needs 8.0. Nor do the bfloat16 cuBLAS variants there: with `VF_CUBLAS_BF16_TC=1`, told 8.0, the model ran at 81.6 and 745 tokens/s. vLLM replays its forward from CUDA graphs, and the variants leave a call captured in a graph to cuBLAS 12.9. [The variants' limits](../../LIMITS.md#the-bf16-cublas-variants)

To keep a model in bfloat16, set `VF_CAPABILITY=8.0 VF_CAPABILITY_ALL=1` through `vf` or the environment of `vf venv` (vLLM reads the capability from NVML, which the launcher answers). Keep vLLM's compilation on, its default. With `--enforce-eager`, or `TORCH_COMPILE_DISABLE=1`, vLLM 0.34.0 stops at its first forward with `CUDA error: named symbol not found`. The layer's line before it names the kernel, `fused_add_rms_norm_kernel<c10::BFloat16, 8, ...>`, and the image, in vLLM's `_C_stable_libtorch.abi3.so`: that kernel is only in the library's machine code for sm_80, and the layer gives that library its sm_75 code. With compilation on, vLLM computes that norm with PyTorch's operations and runs. [The limit](../../LIMITS.md#the-capability-told-with-vf_capability)
