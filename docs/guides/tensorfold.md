# TensorFold on Ubuntu with a Tesla V100

## 1. CUDA

On Ubuntu 24.04 x86-64, check the proprietary NVIDIA R580 driver.

```bash
if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | grep -E 'V100.*, 580[.]'; then
  sudo apt update && sudo apt install -y nvidia-driver-580
fi
```

If you installed the driver, reboot and repeat the check. Building the CUDA kernels requires a CUDA 13 compiler. Step 3 reuses an existing toolkit or installs the compiler in a virtual environment. [Upstream runtime requirements](https://github.com/ashhart/TensorFold/blob/main/RUNBOOK.md)

## 2. VF

The measured server needs VF's shared-memory adaptation from the main branch. Build VF with its standard build targets, then use the launcher from that checkout. [VF shared-memory change](https://github.com/nbald/volta-forever/commit/e8c9774)

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

Build the latest stable TensorFold release with PTX, using the Zig version specified by its source. The precompiled Linux release contains machine code for newer GPUs. [Official TensorFold releases](https://github.com/ashhart/TensorFold/releases/latest)

```sh
mkdir -p ~/tensorfold-v100 && cd ~/tensorfold-v100
vf venv .venv
source .venv/bin/activate
pip install huggingface_hub
export PATH="$PATH:${CUDA_HOME:-/usr/local/cuda}/bin"
if ! nvcc --version 2>/dev/null | grep -q 'release 13[.]'; then
  pip install nvidia-cuda-nvcc nvidia-cuda-runtime nvidia-cuda-cccl
  export PATH="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/nvidia/cu13/bin:$PATH"
fi
tag=$(curl -fsSL https://api.github.com/repos/ashhart/TensorFold/releases/latest | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')
git clone --depth 1 --branch "$tag" https://github.com/ashhart/TensorFold.git && cd TensorFold
version=$(cat .zig-version)
zig="zig-x86_64-linux-$version"
curl -fL "https://ziglang.org/download/$version/$zig.tar.xz" | tar -xJ
NVCC_APPEND_FLAGS='-gencode=arch=compute_80,code=compute_80' \
  "./$zig/zig" build native -Dsm=80 -Dcpu=x86_64_v2 -Dnvcc="$(command -v nvcc)" -Dstrip=true -j2
export PATH="$PWD/zig-out/native/bin:$PATH"
tensorfold-native --version
```

## 4. Test

Download the 20.4 GB MLX checkpoint with 4-bit experts and 8-bit routers. TensorFold 1.3.2 accepts this layout. It refuses GGUF files at startup. [Qwen3.5 checkpoint](https://huggingface.co/henrybravo/Qwen3.5-35B-A3B-4bit)

```sh
hf download henrybravo/Qwen3.5-35B-A3B-4bit --local-dir ~/models/qwen35-4bit
VF_CAPABILITY=8.0 VF_FIT_SHARED=1 TF_QWEN_KV_PAGES=0 \
  vf tensorfold-native serve ~/models/qwen35-4bit --backend cuda \
  --context 512 --parallel 1 --prompt-cache-gib 0 --name qwen --no-thinking
```

When `curl -fsS http://127.0.0.1:8080/health` reports `"backend":"cuda"`, send a request from a second terminal.

```sh
curl -fsS http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"What is the capital of France? Answer with just the city."}],"temperature":0,"max_tokens":32}'
```

The server answered `Paris` and returned `finish_reason: stop`. Repeating the request with `"stop":["Paris"]` returned empty content and `stop`. A separate counting prompt limited to one token returned `one` and `length`.

## 5. Benchmarks

Run the client from a second terminal. It sends one warm-up and three different prompts at temperature zero, with one request at a time. [Benchmark client and prompts](./benchmarks/)

```sh
mkdir -p ~/tensorfold-v100/benchmark && cd ~/tensorfold-v100/benchmark
base=https://raw.githubusercontent.com/nbald/volta-forever/main/docs/guides/benchmarks
curl -fsSLO "$base/tensorfold.py" && curl -fsSLO "$base/prompts.json"
python3 tensorfold.py
```

Each measured request read 227 uncached tokens and generated 64. All three responses began with the requested city. Median time to the first token was 1.344 seconds over loopback. [Measured responses and timings](./benchmarks/tensorfold-results.json)

| Phase | Three runs (tokens/s) | Median (tokens/s) |
| --- | --- | ---: |
| Prefill | 169.57, 169.69, 169.31 | **169.57** |
| Decode | 17.56, 17.10, 16.47 | **17.10** |

Prefill uses TensorFold’s `prefill_seconds`. Decode counts the 63 tokens after the first. On the same V100 and prompts, native llama.cpp reached **460.45 prefill and 99.12 decode tokens/s** using the Q4_K_M GGUF. TensorFold measured **170.35 and 17.18** with MLX affine Q4, or **326.80 and 17.01** with INT8 prompt activations. The two engines use different quantizations of Qwen3.5-35B-A3B. [Same-GPU reference and checkpoint revisions](./benchmarks/tensorfold-results.json)

The INT8 run used `--no-drafts --prompt-precision int8`. All three cities were correct and the generated continuations changed. During BF16 decoding, the CUDA graph replayed successfully and the routed experts’ fused gate/up kernel accounted for about 60% of sampled GPU kernel time. [Option tests and GPU measurements](./benchmarks/tensorfold-results.json)

## 6. Additional notes

The 1.3.2 Linux release contains 131 CUDA bundles with machine code for sm80, sm86, sm89, sm120 and sm121. None embeds PTX. On a Tesla V100-SXM2-32GB with VF 0.1.1, the released server returned `CUDA_ERROR_NO_BINARY_FOR_GPU (209)` for both a Qwen3.5 MLX checkpoint and NVIDIA’s Qwen3.6 NVFP4 checkpoint. The default, 8.0 and 12.0 capability profiles gave the same result. [TensorFold 1.3.2 release](https://github.com/ashhart/TensorFold/releases/tag/v1.3.2)

On the V100, both 4-bit prompt kernels reproduced all 132,096 BF16 reference values for the first layer’s shared-expert gate projection using real weights from the checkpoint. The causal attention kernel matched a separate CPU reference across 68,608 outputs with a maximum absolute error of 0.00133. [TensorFold 4-bit prompt kernels](https://github.com/ashhart/TensorFold/blob/v1.3.2/zig/kernels/cuda/qmm_prefill.cu)

The NVIDIA Qwen3.6 NVFP4 path additionally uses Ampere memory barriers. With experimental VF support, three TensorFold prompt kernels reproduced 306,432 CPU results exactly using a slice of the checkpoint’s layer-zero gate and up weights. [NVIDIA checkpoint](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) · [TensorFold prompt kernel](https://github.com/ashhart/TensorFold/blob/v1.3.2/zig/kernels/cuda/prompt16_lean.cuh)
