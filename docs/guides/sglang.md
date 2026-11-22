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

Build VF from the main branch for the 8.0 capability profile used below. VF translates the PTX retained in SGLang’s CUDA kernels for the V100.

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

Start the official SGLang server in the active environment on a 32 GiB V100. It downloads the 4-bit AWQ checkpoint on first use. The checkpoint keeps attention projections in floating point. SGLang's model override names those projections so its WNA16 loader can initialize the attention layers. [Tested checkpoint](https://huggingface.co/QuantTrio/Qwen3.5-35B-A3B-AWQ/tree/2f4fac011fc06e736782b0b59394577790da7d79)

```sh
export PATH="$CUDA_HOME/bin:$PATH"
export PTXAS_OPTIONS='--override-directive-values --maxrregcount=192 --opt-level=3'
export VF_CAPABILITY=8.0
export SGLANG_MOE_CONFIG_DIR="$HOME/volta-forever/tuning/sglang"
export NVCC_APPEND_FLAGS='-gencode=arch=compute_80,code=compute_80'
export SGLANG_MAMBA_CONV_DTYPE=float16 FLASHINFER_USE_CUDA_NORM=1
export SGLANG_EXPERIMENTAL_LORA_OPTI=1 SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN=1
export SGLANG_OPT_LORA_FUSED_TOPK_PACK=0
export SGLANG_ENABLE_METRICS_DEVICE_TIMER=1
export OMP_NUM_THREADS=4 MAX_JOBS=4
awq='{"quantization_config":{"quant_method":"awq","bits":4,"group_size":128,"version":"gemm","zero_point":true,"modules_to_not_convert":["visual","linear_attn","self_attn.qkv_proj","self_attn.o_proj","shared_expert","mlp.gate","model.layers.0.","mtp"]}}'
python -m sglang.launch_server --model-path QuantTrio/Qwen3.5-35B-A3B-AWQ \
  --served-model-name qwen --dtype float16 --quantization moe_wna16 \
  --json-model-override-args "$awq" --attention-backend triton --sampling-backend pytorch \
  --context-length 512 --max-running-requests 1 --chunked-prefill-size 32 \
  --mem-fraction-static 0.85 --model-loader-extra-config '{"enable_multithread_load":false}' \
  --cuda-graph-backend-prefill disabled --cuda-graph-bs-decode 1 --cuda-graph-backend-decode full \
  --enable-metrics --skip-server-warmup --watchdog-timeout 3600 \
  --host 127.0.0.1 --port 8000
```

The first request compiles kernels. From a second terminal, send the question below. The V100 returned **Paris** in two output tokens.

```sh
curl -fsS http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"What is the capital of France? Answer with just the city."}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

## 5. Benchmarks

With the server running, run `python3 docs/guides/benchmarks/sglang.py` from a checkout of this project. The client sends one warm-up and three distinct requests. Each measured request processed 227 uncached input tokens and generated 64 output tokens. SGLang’s GPU forward timers cover all 227 prefill tokens and the 63 output tokens after the first. [Client and prompts](./benchmarks/)

The command above measured **225.57 tokens/s in prefill** and **52.99 tokens/s in decode**. The SGLang configuration directory selects the W4A16 kernel tile sizes for this V100. The same model with SGLang’s default tiles measured 219.92 and 45.01 tokens/s. [Per-request results](./benchmarks/sglang-stock-sm80.json)

An earlier VF 0.1.1 run at capability 7.5 measured **181.48 prefill** and **38.83 decode** tokens/s at 32 tokens per chunk. At eight tokens per chunk we compared the effect of decode graphs on the same card. [Earlier 32-token results](./benchmarks/sglang-stock-compile.json)

| Decode CUDA graph | Median prefill (tokens/s) | Median decode (tokens/s) |
| --- | ---: | ---: |
| Enabled | 62.55 | 38.25 |
| Disabled | 72.97 | 12.27 |

The two servers ran in sequence on the same V100. Changing `--cuda-graph-backend-decode full` to `disabled` was the only configuration change. Decode ran **3.12× faster** with CUDA graphs. [Paired results](./benchmarks/sglang-stock-graphs.json) · [Initial eager baseline](./benchmarks/sglang-stock-results.json)

## 6. Additional notes

The 8.0 run used VF main at `883e420`, SGLang 0.5.23, PyTorch 2.14.1+cu130, CUDA toolkit 13.5 and driver 580.178.04. Triton selects FP16 tensor-core operations for the quantized experts at this capability. The experimental LoRA switches select SGLang’s JIT expert router and standard top-k layout. [Tested VF source](https://github.com/nbald/volta-forever/commit/883e4209021f411c5fadd67b9313717cf60409a1)

SGLang also answered **Paris** with the checkpoint’s own prediction head and four speculative steps (NEXTN). A VF source build at `a381fc7` measured **171.66 tokens/s in prefill** and **37.60 in decode**. Decode includes draft generation, target verification and draft extension. [NEXTN settings and per-request results](./benchmarks/sglang-stock-mtp.json)
