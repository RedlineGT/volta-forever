# ZML on a Tesla V100

## 1. CUDA

On Ubuntu 24.04 x86-64, run this check for the proprietary NVIDIA R580 driver.

```bash
if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | grep -E 'V100.*, 580[.]'; then
  sudo apt update && sudo apt install -y nvidia-driver-580
fi
```

If you installed the driver, reboot and repeat the check. It should print your V100 and driver version.

Bazel downloads the CUDA 13 libraries for the source example. The 35B LLMD server also needs CUDA 12.9.2 `ptxas`, included in the VF source setup below.

## 2. VF

For the 35B server, build VF with support for XLA’s external assembler. The source setup includes the CUDA libraries and the CUDA 12.9.2 assembler. [VF compiler support](https://github.com/nbald/volta-forever/commit/9013f262ee3866d999013736f6886429eb4803c4)

```bash
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
sudo apt install -y git unzip zstd
git clone --depth 1 https://github.com/zml/zml.git
cd zml
./bazel.sh build -c opt --@zml//platforms:cuda=true //examples/llm
```

## 4. Test

```sh
vf ./bazel-bin/examples/llm/llm --model=hf://Qwen/Qwen3.5-0.8B \
  --seqlen=128 --topk=1 --prompt="The capital of France is"
```

Look for `platform: cuda` and `Tesla V100` in the device listing. The tested model answered **Paris**.

## 5. Benchmarks

The 35B benchmark uses ZML’s official Linux LLMD server. Upstream distributes its CUDA executable and runfiles together in a registry image. These commands extract the measured September 24 distribution into your home directory and run it directly on Ubuntu. Allow 4.7 GB for LLMD before downloading the model. [Upstream distribution](https://zml.ai/llmd/) · [crane export](https://github.com/google/go-containerregistry/blob/main/cmd/crane/doc/crane_export.md)

```bash
(
set -euo pipefail
mkdir -p ~/llmd-v100 && cd ~/llmd-v100
curl -fL --retry 3 -o crane.tar.gz https://github.com/google/go-containerregistry/releases/latest/download/go-containerregistry_Linux_x86_64.tar.gz
tar --no-same-owner -xzf crane.tar.gz crane
record="$HOME/volta-forever/docs/guides/benchmarks/zml-llmd-bf16-product.json"
image=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["engine"]["image"])' "$record")
./crane export --platform linux/amd64 "$image" - | tar --no-same-owner -xf - llmd
./llmd/llmd --help
)
```

Download the AMD MXFP4 checkpoint once. The revision below selects the exact 14 weight shards used in the benchmark. [Checkpoint files](https://huggingface.co/amd/Qwen3.5-35B-A3B-MXFP4/tree/2e19c6576db91e5d5a93455415619262218bf8a1) · [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

```sh
export MODEL="$HOME/models/Qwen3.5-35B-A3B-MXFP4"
vf venv "$HOME/.venvs/zml-download"
"$HOME/.venvs/zml-download/bin/pip" install huggingface_hub
"$HOME/.venvs/zml-download/bin/hf" download amd/Qwen3.5-35B-A3B-MXFP4 \
  --revision 2e19c6576db91e5d5a93455415619262218bf8a1 --local-dir "$MODEL"
```

Start the extracted server. The measured configuration uses XLA autotuning, tree accumulation and single-token decoding. The two BF16 multiply measurements below both use `tree`. With `VF_BF16_TC_TILES=1` and `VF_BF16_COLUMN=tree`, results are not bit-reproducible from run to run when a kernel leaves unused operands of a product uninitialised; the differences are of rounding size. [Determinism of the BF16 switches](../emulations.md#4-bf16)

```bash
cd ~/llmd-v100/llmd
export VF_BF16_COLUMN=tree
VF_CAPABILITY=8.0 VF_BF16_TC_TILES=1 VF_PROXY_CACHE=0 \
  XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1 \
  XDG_CACHE_HOME="$(mktemp -d)" CUDA_CACHE_PATH="$(mktemp -d)" \
  vf ./llmd --model="${MODEL:?Set MODEL to the local AMD checkpoint directory}" --model-name=qwen \
  --backend=triton --batch-size=1 --prefill-chunk-size=256 --token-batch-size=256 \
  --page-chunk-size=16 --max-context-len=512 --gpu-memory-fraction=0.94 \
  --cache-memory-fraction=0.97 --topk=1 --listen=127.0.0.1:8000
```

In a second terminal, run the client against the idle server. The client prints `decode_tokens_per_s` for the visible tokens after the first and `decode_steps_per_s` for LLMD’s internal steps. To repeat the before/after comparison, build VF at `f2326f6` and `c6f678c` in turn, restarting the server with the same options and fresh caches on the same card. For the separate GPU timings, start a fresh server with the CUPTI collector. [Common prompts](./benchmarks/prompts.json) · [CUPTI commands](./benchmarks/README.md#llmd-gpu-prefill)

```bash
python3 "$HOME/volta-forever/docs/guides/benchmarks/zml.py"
```

After a Lisbon warm-up, each request used a distinct 227-token prompt and produced 64 tokens. All replies named the expected city and object. The table compares the general BF16 multiply optimization on the same V100 with tree accumulation. The GPU prefill column comes from a separate CUPTI run.

| VF build | GPU prefill (tokens/s) | Prompt tokens / server TTFT (tokens/s) | Decode after first token (tokens/s) |
| --- | ---: | ---: | ---: |
| Before the BF16 change | 746.45 | 725.24 | 48.55 |
| With the BF16 change | **774.63** | **751.66** | **50.16** |

Each reply contributes 63 visible tokens after the first. LLMD counted 64 internal steps per measured reply in this pair, giving 49.32 and 50.95 internal steps/s respectively. The decode column uses 63 divided by the server’s decode time for each request. Native 1Cat-vLLM on this card reached 1,443.33 / 76.09 with a different checkpoint, QuantTrio AWQ. [Per-request replies, counters and graph spans](./benchmarks/zml-llmd-bf16-product.json)

An independent run using the extracted distribution on a second V100 returned Paris and completed all three benchmark requests. It measured **751.66 prompt tokens/s through the first token** and **49.53 visible decode tokens/s**. [Extracted-bundle timings](./benchmarks/zml-llmd-bf16-product.json)

Tree accumulation saved 1.87 ms per decode graph in an earlier comparison with the column optimization off. Its FP32 sums can differ under cancellation. Ordered accumulation matched scalar bits on all 16,384 operator outputs tested. [Column results](./benchmarks/zml-llmd-column.json) · [Tile comparison](./benchmarks/zml-llmd-autotune.json) · [Chunk-size comparison](./benchmarks/zml-llmd-mxfp4.json)

## 6. Additional notes

The source example generated **Paris** with Qwen3.5-0.8B. Its 35B AWQ loader expects fused expert weights that the QuantTrio checkpoint stores separately. The earlier AWQ experiment used a modified loader. The 35B MXFP4 results above use the original LLMD executable. [ZML source example](https://github.com/zml/zml/commit/f430c3530fbc6d99f734b641eb8f3fd699868843) · [Experimental AWQ results](./benchmarks/zml-awq-experimental.json)

The October 23 LLMD build stopped at a bundled sm_120 CUTLASS expert kernel on V100. Use the executable together with the runfiles from its own distribution. [Upstream expert library](https://github.com/zml/flashinfer/releases/tag/cutlass-moe-v0.2.0)

Since VF 0.1.3, with `VF_CAPABILITY=8.0`, `libzml_cuda.so` is still told 8.0, because its kernels are PTX, and a cuDNN whose kernels run from the translation of their sm_75 machine code is told 7.5, with one line in the log (the cuDNN 9.24 that `tests/gpu/zml.sh` runs is). The measurements above were made with earlier builds, which told cuDNN 8.0 as well. [Capability by caller](../architecture.md#4-what-a-program-is-told-about-the-gpu)
