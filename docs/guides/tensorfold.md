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

Download the 20.4 GB MLX checkpoint with 4-bit experts and 8-bit routers. TensorFold 1.3.2 and 1.3.3 accept this layout. It refuses GGUF files at startup. [Qwen3.5 checkpoint](https://huggingface.co/henrybravo/Qwen3.5-35B-A3B-4bit)

```sh
hf download henrybravo/Qwen3.5-35B-A3B-4bit --local-dir ~/models/qwen35-4bit
VF_CAPABILITY=8.0 VF_FIT_SHARED=1 VF_BF16_TC_GROUPS=1 VF_BF16_ROW=tree TF_QWEN_KV_PAGES=0 \
  vf tensorfold-native serve ~/models/qwen35-4bit --backend cuda \
  --context 512 --parallel 1 --prompt-cache-gib 0 --name qwen --no-thinking \
  --no-drafts --prompt-precision int8
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

The three measured requests each read 227 uncached tokens and generated 64. The medians use TensorFold’s native phase timers and exclude the warm-up. Decode counts the 63 tokens after the first. Both TensorFold runs used the same official 1.3.3 executable, MLX checkpoint and V100-SXM2-32GB with drafting disabled.

| Build | Median prefill (tokens/s) | Median decode (tokens/s) |
| --- | ---: | ---: |
| TensorFold before the four-lane projection change | 463.81 | 70.78 |
| TensorFold with VF from section 4 | **463.95** | **72.36** |
| Native llama.cpp with GGUF Q4_K_M on the same card | 474.57 | 99.77 |

Four warp lanes share the work on four expert output columns. Across the 40 layers, the gate and up projections took 2.15 ms per generated token, down from 2.48 ms. Decode improved by 2.24% with identical replies to all four prompts. Earlier paired runs measured 57.69 to 69.18 tokens/s for the single-row specialization and 69.26 to 70.31 for the Q4 weight lookup. [TensorFold measurements](./benchmarks/tensorfold-affine-q4.json) · [Same-card native reference](./benchmarks/llamacpp-reference.json)

Earlier measurements cover shared operand checks, dense projections, FP32 operand forwarding and two- and four-product groups. [Shared operand checks](./benchmarks/tensorfold-expert-checks.json) · [Dense projections](./benchmarks/tensorfold-dense-bf16.json) · [Operand forwarding](./benchmarks/tensorfold-bf16-forwarding.json) · [Four-product groups](./benchmarks/tensorfold-four-products.json) · [Two-product groups](./benchmarks/tensorfold-bf16-row.json)

## 6. Additional notes

A separate experiment with four local changes to TensorFold’s CUDA kernels reached 649.94 tokens/s prefill and 109.25 decode on another V100. Its native llama.cpp reference reached 474.57 and 99.77. [Experimental build and comparison](./benchmarks/tensorfold-experimental.json)

TensorFold 1.3.3 selects fixed CUDA kernels for MLX affine Q4 experts. Its `TF_NVFP4_DECODE_WARPS` setting applies to NVFP4 weights, and its experimental `TF_CUDA_TEST_MLP128` setting requires sm120. The four local CUDA changes in that separate experiment require rebuilding the modified engine. [Official expert dispatch](https://github.com/ashhart/TensorFold/blob/v1.3.3/zig/src/cuda/experts.zig#L143-L223) · [Official tile override](https://github.com/ashhart/TensorFold/blob/v1.3.3/zig/src/cuda/sm120_test_override.zig#L6-L13)

The official Linux binaries contain 131 CUDA bundles in 1.3.2 and 139 in 1.3.3. Both ship machine code for sm80, sm86, sm89, sm120 and sm121, with zero embedded PTX images. Both binaries returned `CUDA_ERROR_NO_BINARY_FOR_GPU (209)` on the V100 with the Qwen3.5 MLX checkpoint. With release 1.3.2, the default, 8.0 and 12.0 capability profiles also gave this error with NVIDIA’s Qwen3.6 NVFP4 checkpoint. [TensorFold 1.3.2 release](https://github.com/ashhart/TensorFold/releases/tag/v1.3.2) · [TensorFold 1.3.3 release](https://github.com/ashhart/TensorFold/releases/tag/v1.3.3)

On the V100, both 4-bit prompt kernels reproduced all 132,096 BF16 reference values for the first layer’s shared-expert gate projection using real weights from the checkpoint. The causal attention kernel matched a separate CPU reference across 68,608 outputs with a maximum absolute error of 0.00133. [TensorFold 4-bit prompt kernels](https://github.com/ashhart/TensorFold/blob/v1.3.2/zig/kernels/cuda/qmm_prefill.cu)

The NVIDIA Qwen3.6 NVFP4 path additionally uses Ampere memory barriers. With experimental VF support, three TensorFold prompt kernels reproduced 306,432 CPU results exactly using a slice of the checkpoint’s layer-zero gate and up weights. [NVIDIA checkpoint](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) · [TensorFold prompt kernel](https://github.com/ashhart/TensorFold/blob/v1.3.2/zig/kernels/cuda/prompt16_lean.cuh)
