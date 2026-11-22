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

The 35B benchmark uses the official Linux LLMD executable and its adjacent `llmd.runfiles` directory. The measured build is the September 24 distribution. Sections 3 and 4 run ZML’s separate source example. [Upstream LLMD distribution](https://zml.ai/llmd/)

Download the AMD MXFP4 checkpoint once. The revision below selects the exact 14 weight shards used in the benchmark. [Checkpoint files](https://huggingface.co/amd/Qwen3.5-35B-A3B-MXFP4/tree/2e19c6576db91e5d5a93455415619262218bf8a1) · [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

```sh
export MODEL="$HOME/models/Qwen3.5-35B-A3B-MXFP4"
vf venv "$HOME/.venvs/zml-download"
"$HOME/.venvs/zml-download/bin/pip" install huggingface_hub
"$HOME/.venvs/zml-download/bin/hf" download amd/Qwen3.5-35B-A3B-MXFP4 \
  --revision 2e19c6576db91e5d5a93455415619262218bf8a1 --local-dir "$MODEL"
```

From the directory containing `llmd` and `llmd.runfiles`, start the server. The measured configuration uses XLA autotuning, tree accumulation and single-token decoding. To reproduce the other rows below, use `ordered` for scalar-order FP32 accumulation or `0` to disable the single-column optimization.

```bash
export VF_BF16_COLUMN=tree
VF_CAPABILITY=8.0 VF_BF16_TC_TILES=1 VF_PROXY_CACHE=0 \
  XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1 \
  XDG_CACHE_HOME="$(mktemp -d)" CUDA_CACHE_PATH="$(mktemp -d)" \
  vf ./llmd --model="${MODEL:?Set MODEL to the local AMD checkpoint directory}" --model-name=qwen \
  --backend=triton --batch-size=1 --prefill-chunk-size=256 --token-batch-size=256 \
  --page-chunk-size=16 --max-context-len=512 --gpu-memory-fraction=0.94 \
  --cache-memory-fraction=0.97 --topk=1 --listen=127.0.0.1:8000
```

In a second terminal, run the client against the idle server. For a paired comparison, stop LLMD, change `VF_BF16_COLUMN`, repeat the server command on the same card and rerun the client. For the separate GPU timings, start a fresh server with the CUPTI collector. [Common prompts](./benchmarks/prompts.json) · [CUPTI commands](./benchmarks/README.md#llmd-gpu-prefill)

```bash
python3 "$HOME/volta-forever/docs/guides/benchmarks/zml.py"
```

Each median covers three uncached 227-token prompts and 64-token replies after warm-up. All replies began with the expected city. LLMD times 63 decode steps after the first token. The GPU columns use a separate series of marked requests and time the complete forward graphs. [Per-request replies, counters and graph spans](./benchmarks/zml-llmd-column.json)

| Single-card pair | Column mode | GPU prefill (tokens/s) | GPU decode (graphs/s) | Prompt tokens / server TTFT (tokens/s) | Server decode (steps/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| V100 `08cd61f4` | Off → ordered | 744.01 → 750.70 | 49.58 → 50.82 | 737.01 → 746.71 | 45.94 → **47.18** |
| V100 `148f2829` | Off → tree | 748.68 → 741.99 | 49.18 → 54.17 | 737.01 → 737.01 | 45.57 → **49.19** |

Tree saved **1.87 ms per decode graph**, including **1.72 ms** in the two expert projections. Its reassociated FP32 sums can differ under cancellation. Ordered matched scalar bits on all 16,384 operator outputs tested. A clean Ubuntu 24.04 run on a third V100 reproduced **750.11 GPU prefill tokens/s** with ordered. Native 1Cat-vLLM reached **1427.41 / 76.13** and **1443.33 / 76.09** prefill/decode on the two respective cards with QuantTrio AWQ. Both LLMD pairs remain below those reference rates, which use a different Q4 checkpoint and server phase timers. [Column results and numerical limits](./benchmarks/zml-llmd-column.json) · [Earlier two-by-two tile comparison](./benchmarks/zml-llmd-autotune.json) · [Earlier chunk-size comparison](./benchmarks/zml-llmd-mxfp4.json)

## 6. Additional notes

The source example generated **Paris** with Qwen3.5-0.8B. Its 35B AWQ loader expects fused expert weights that the QuantTrio checkpoint stores separately. The earlier AWQ experiment used a modified loader. The 35B MXFP4 results above use the original LLMD executable. [ZML source example](https://github.com/zml/zml/commit/f430c3530fbc6d99f734b641eb8f3fd699868843) · [Experimental AWQ results](./benchmarks/zml-awq-experimental.json)

The October 23 LLMD build stopped at a bundled sm_120 CUTLASS expert kernel on V100. Use the executable together with the runfiles from its own distribution. [Upstream expert library](https://github.com/zml/flashinfer/releases/tag/cutlass-moe-v0.2.0)
