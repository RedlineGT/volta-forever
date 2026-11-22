# SGLang on a Tesla V100

## 1. CUDA

On Ubuntu 24.04 x86-64, run this check for the proprietary NVIDIA R580 driver.

```bash
if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | grep -E 'V100.*, 580[.]'; then
  sudo apt update && sudo apt install -y nvidia-driver-580
fi
```

If you installed the driver, reboot and repeat the check. It should print your V100 and driver version.

SGLang compiles FlashInfer kernels on first launch. Use a complete CUDA 13 toolkit so the compiler and headers come from the same installation. This block installs the toolkit only when CUDA 13 is missing. [NVIDIA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#ubuntu)

```bash
export PATH="$PATH:${CUDA_HOME:-/usr/local/cuda}/bin"
if ! nvcc --version 2>/dev/null | grep -q 'release 13[.]'; then
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

Build VF from the main branch for the 8.0 capability profile used below. VF translates SGLang’s CUDA kernels and the Marlin expert reduction for the V100.

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
mkdir -p ~/sglang-v100 && cd ~/sglang-v100
vf venv .venv
source .venv/bin/activate
pip install sglang
```

## 4. Test

Start SGLang in the active environment. It downloads the 4-bit AWQ checkpoint and selects Marlin for its quantized experts. [Tested checkpoint](https://huggingface.co/QuantTrio/Qwen3.5-35B-A3B-AWQ/tree/2f4fac011fc06e736782b0b59394577790da7d79)

```sh
export PATH="$CUDA_HOME/bin:$PATH"
export VF_CAPABILITY=8.0
export NVCC_APPEND_FLAGS='-gencode=arch=compute_80,code=compute_80'
export SGLANG_MAMBA_CONV_DTYPE=float16 FLASHINFER_USE_CUDA_NORM=1
export SGLANG_EXPERIMENTAL_LORA_OPTI=1 SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN=1
export SGLANG_OPT_LORA_FUSED_TOPK_PACK=0
export SGLANG_ENABLE_METRICS_DEVICE_TIMER=1
export OMP_NUM_THREADS=4 MAX_JOBS=4
python -m sglang.launch_server --model-path QuantTrio/Qwen3.5-35B-A3B-AWQ \
  --served-model-name qwen --dtype float16 --context-length 512 \
  --max-running-requests 1 --mem-fraction-static 0.85 --enable-metrics \
  --attention-backend triton --sampling-backend pytorch --chunked-prefill-size 256 \
  --cuda-graph-backend-prefill breakable --cuda-graph-bs-prefill 8 16 32 64 128 192 256 \
  --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 \
  --host 127.0.0.1 --port 8000
```

Once SGLang reports that the server is ready, use a second terminal to send the question below. The V100 returned **Paris** in two output tokens.

```sh
curl -fsS http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"What is the capital of France? Answer with just the city."}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

## 5. Benchmarks

With the server running, send one warm-up and three distinct requests from the second terminal.

```sh
python3 ~/volta-forever/docs/guides/benchmarks/sglang.py
```

The client checks SGLang’s phase timers and token counts. Each measured request processed 227 uncached input tokens and generated 64 output tokens. SGLang’s GPU forward timers cover all 227 prefill tokens and the 63 output tokens after the first. [Client and prompts](./benchmarks/)

The command above measured **2,015.34 tokens/s in prefill** and **77.36 tokens/s in decode**. Each 227-token prompt ran in a single CUDA graph. Native 1Cat-vLLM measured 1,427.41 and 76.13 on the same physical V100, checkpoint and prompts. Omitting `--attention-backend triton --sampling-backend pytorch` selects FlashInfer and measured 1,995.78 prefill and 75.15 decode tokens/s on the same V100. [SGLang measurements](./benchmarks/sglang-stock-marlin.json) · [1Cat reference](./benchmarks/vllm-expert-tiles.json)

In a separate paired run on the same V100, Marlin measured 2,001.75 prefill and 77.39 decode tokens/s at a 256-token chunk, and 1,565.10 and 77.38 at a 128-token chunk. Native 1Cat-vLLM on that card measured 1,361.25 and 76.31. The earlier Triton WNA16 expert backend measured 311.97 and 52.68 tokens/s at a 32-token chunk. [Earlier WNA16 results](./benchmarks/sglang-stock-sm80.json)

## 6. Additional notes

The run used VF main at `d9f8083`, SGLang 0.5.23, PyTorch 2.14.1+cu130, CUDA 13.5 and the NVIDIA 580.178.04 driver. The stock wheel contains an Ampere binary for the expert reduction. VF translates it to Volta. The LoRA settings select SGLang's JIT expert router and the three-field output expected by the floating-point layer. FlashInfer uses its CUDA normalization kernel. [Tested VF source](https://github.com/nbald/volta-forever/commit/d9f8083afe412da84e016040acb5424e9fde3602)

SGLang also answered **Paris** with the checkpoint’s own prediction head and four speculative steps (NEXTN). A VF source build at `a381fc7` measured **171.66 tokens/s in prefill** and **37.60 in decode**. Decode includes draft generation, target verification and draft extension. [NEXTN settings and per-request results](./benchmarks/sglang-stock-mtp.json)
