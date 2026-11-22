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

VF 0.1.2 supports the 8.0 capability profile used below. VF translates SGLang’s CUDA kernels and the Marlin expert reduction for the V100.

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
export SGLANG_MAMBA_CONV_DTYPE=float16
export FLASHINFER_USE_CUDA_NORM=1
export SGLANG_ENABLE_METRICS_DEVICE_TIMER=1
export OMP_NUM_THREADS=4 MAX_JOBS=4
python -m sglang.launch_server --model-path QuantTrio/Qwen3.5-35B-A3B-AWQ \
  --revision 2f4fac011fc06e736782b0b59394577790da7d79 \
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

After one warm-up, each request processed 227 uncached input tokens and generated 64. The first token comes from prefill. SGLang’s GPU timers measure prefill over all 227 inputs and decode over the remaining 63 outputs.

| Server on the same V100 and AWQ checkpoint | Median prefill (tokens/s) | Median decode (tokens/s) |
|---|---:|---:|
| SGLang with the command above | **2,016.06** | **77.33** |
| SGLang with FlashInfer’s default CuTe normalization | 1,988.55 | 74.62 |
| Native 1Cat-vLLM | 1,427.41 | 76.13 |

Each prompt fit in a single CUDA graph. The server answered Paris and returned the expected city and notebook details for Kyoto, Lyon and Lima. [Phase timers and answers](./benchmarks/sglang-default-operators.json) · [Native reference](./benchmarks/vllm-expert-tiles.json)

## 6. Additional notes

SGLang’s upstream AWQ CUDA kernels require at least compute capability 7.5. A native V100 reports 7.0. [Upstream AWQ requirement](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/layers/quantization/awq/awq.py#L100-L108)

The run used VF main at `1740d8f`, SGLang 0.5.23, PyTorch 2.14.1+cu130, CUDA 13.5 and the NVIDIA 580.178.04 driver. VF translates the stock wheel’s Ampere kernels for expert routing and reduction, and FlashInfer’s default CuTe normalization kernels, for the V100. [Tested VF source](https://github.com/nbald/volta-forever/commit/1740d8fc1f6291b5838112b2d17d7d9e10dd787c)

The launch command selects FlashInfer’s CUDA normalization on V100. To reproduce the CuTe result, unset `FLASHINFER_USE_CUDA_NORM` before starting the server. Both runs use the wheel’s default expert router.

The optional JIT router with FlashInfer’s CUDA normalization measured 2,002.22 prefill and 77.20 decode tokens/s in the same virtual environment. [Installation and comparison](./benchmarks/sglang-fresh-venv.json)

For tensor parallelism, fetch the CUDA 12 build of NCCL and add `--tp 2` to the server command in section 4. (Any use of several GPUs from PyTorch needs that build, `torch.distributed` and `torch.nn.DataParallel` too: [Several GPUs and NCCL](../../LIMITS.md#several-gpus-and-nccl).) SGLang 0.5.23 could not parse the GPU UUIDs in `CUDA_VISIBLE_DEVICES` for its optional custom all-reduce. The two workers used the automatically selected CUDA 12 NCCL 2.32.6 transport over NVLink (P2P IPC). On two V100s, the same AWQ checkpoint answered Paris and all four benchmark prompts, with medians of **2819.81 prefill tokens/s** and **86.37 decode steps/s**. The client uses the longest cumulative GPU duration across the ranks for each whole phase. Its separate median time to first visible text was **95.34 ms**. Four V100s passed the separate NCCL reduction test. The four-GPU SGLang startup stopped before generation. [Per-rank durations, client timings and replies](./benchmarks/sglang-tensor-parallel.json) · [Upstream issue and other multi-GPU results](https://github.com/nbald/volta-forever/issues/2)

```sh
vf fetch --only libnccl
```

After the server starts with two ranks, run the client from a second terminal.

```sh
python3 ~/volta-forever/docs/guides/benchmarks/sglang.py --tp-size 2
```
