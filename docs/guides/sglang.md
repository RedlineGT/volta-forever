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

The installer fetches the CUDA 12.9.2 cuBLAS libraries used by `vf`. If an existing installation lacks them, run `vf fetch`.

```sh
sudo apt update
sudo apt install -y ca-certificates curl python3 python3-venv xz-utils
curl -fsSLo /tmp/vf-install.sh https://raw.githubusercontent.com/nbald/volta-forever/main/install.sh
bash /tmp/vf-install.sh
export PATH="$HOME/.local/bin:$PATH"
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
export NVCC_APPEND_FLAGS='-gencode=arch=compute_75,code=compute_75'
export SGLANG_MAMBA_CONV_DTYPE=float16 FLASHINFER_USE_CUDA_NORM=1
export SGLANG_EXPERIMENTAL_LORA_OPTI=1 SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN=1
export SGLANG_OPT_LORA_FUSED_TOPK_PACK=0
export TORCH_COMPILE_DISABLE=1 SGLANG_ENABLE_METRICS_DEVICE_TIMER=1
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

The 32-token configuration above measured **180.82 tokens/s in prefill** and **38.12 tokens/s in decode**. SGLang uses its own JIT expert router and standard top-k layout. [32-token results](./benchmarks/sglang-stock-jit32.json)

To isolate CUDA graphs, we also compared the same two launches at a smaller eight-token chunk size.

| Decode CUDA graph | Median prefill (tokens/s) | Median decode (tokens/s) |
| --- | ---: | ---: |
| Enabled | 62.55 | 38.25 |
| Disabled | 72.97 | 12.27 |

The two servers ran in sequence on the same V100. Changing `--cuda-graph-backend-decode full` to `disabled` was the only configuration change. Decode ran **3.12× faster** with CUDA graphs. [Paired results](./benchmarks/sglang-stock-graphs.json) · [Initial eager baseline](./benchmarks/sglang-stock-results.json)

## 6. Additional notes

The stock run used VF 0.1.1, SGLang 0.5.23, PyTorch 2.14.1+cu130, CUDA toolkit 13.5 and driver 580.178.04. The command selects WNA16 at VF’s default capability of 7.5. The experimental LoRA configuration switches select the wheel’s CUDA JIT expert router for 32-token chunks. The run used the checkpoint’s own weights.

An earlier experiment with an SGLang adapter, larger chunks and CUDA graph decode measured 643.91 tokens/s in prefill and 67.30 in decode. [Archived measurements](./benchmarks/sglang-results.json) · [Archived configuration](https://github.com/nbald/volta-forever/blob/7ead7334d0909483b63139d253aead22cf97820a/docs/guides/sglang.md)
