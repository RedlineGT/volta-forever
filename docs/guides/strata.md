# Run Strata on a V100

## 1. CUDA

On Ubuntu 24.04 x86-64, run this check for the proprietary NVIDIA R580 driver.

```bash
if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | grep -E 'V100.*, 580[.]'; then
  sudo apt update && sudo apt install -y nvidia-driver-580
fi
```

If you installed the driver, reboot and repeat the check. It should print your V100 and driver version.

Strata ships the CUDA runtime and cuBLAS with its engine. The host can run this demo without installing a CUDA toolkit.

## 2. VF

```sh
sudo apt update && sudo apt install -y ca-certificates curl python3 xz-utils
curl -fsSL https://raw.githubusercontent.com/nbald/volta-forever/main/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
vf doctor
```

## 3. Engine

Install the upstream terminal, then use its signed update feed to install the CUDA 13 engine. [Strata installation](https://github.com/Niko1221/Strata#install)

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install-strata.sh
sh install-strata.sh
strata --version
STRATA_CUDA=13 vf strata update --backend cuda --no-control
```

## 4. Test

Download the 22.85 GB Qwen3.5-35B-A3B MTP GGUF. Strata calls the compatible engine profile `qwen36`. The explicit file path selects the Qwen3.5 weights. [Checkpoint](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-MTP-GGUF)

```sh
model=Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
curl -fL --retry 3 -o "$model" \
  "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-MTP-GGUF/resolve/main/$model"
CUDA_MODULE_LOADING=LAZY STRATA_CUDA=13 vf strata launch --family qwen36 --model UD-Q4_K_XL \
  --model-path "$PWD/$model" --context 8k --vision off --tune skip --set STRATA_Q36_BLAS_F32=1 \
  --set prompt-cache=0 --set conversation-cache-mib=0 --access local --port 9085 \
  --no-api-key --yes --no-tui
```

Ask “What is the capital of France? Answer in one word.” in the local chat. The unpinned installation selected the signed Strata 0.2.18 terminal and CUDA 13 engine 0.1.64. It answered **Paris** on a Tesla V100-SXM2-32GB with driver 580.178.04 and VF 0.1.1.

## 5. Benchmarks

With that server running, download the client and prompts to the same directory. The client runs a warm-up and three different chats at temperature zero with thinking disabled.

```sh
base=https://raw.githubusercontent.com/nbald/volta-forever/main/docs/guides/benchmarks
curl -fsSLO "$base/strata-ninfer.py"
curl -fsSLO "$base/prompts.json"
python3 strata-ninfer.py strata
```

The command above enables cuBLAS for the prompt’s FP32 router and GDN projections. We tested the setting on one Tesla V100-SXM2-32GB, alternating the default and cuBLAS paths in four blocks. Both ran Strata 0.1.64 CUDA 13 with VF 0.1.1, the same MTP checkpoint and five rounds of three chats. [Upstream implementation](https://github.com/Niko1221/Strata/blob/v0.2.18/engine/src/qwen35/q36_engine.cu#L1655-L1683)

| FP32 prompt projections | Median prefill (range) | Median decode (range) |
| --- | ---: | ---: |
| Default | 1,869.9 (1,865.2–1,872.9) tok/s | 152.4 (141.7–158.8) tok/s |
| cuBLAS, as launched above | **1,894.8** (1,877.6–1,901.2) tok/s | **152.5** (145.8–158.7) tok/s |

Each of the 30 measured requests processed 227 uncached tokens, generated 64 and began with the requested city. On a second V100 of the same type, native Strata CUDA 12 measured **1,872.9 prefill / 151.1 decode tok/s** on 0.1.63 and **1,877.6 / 150.9** on 0.1.64. The results file includes the individual replies, phase timers, both native comparisons and the chunk-size and expert-routing trials. [Full measurements](./benchmarks/strata-results.json)

## 6. Additional notes

The standard Qwen3.5-35B-A3B Q4_K_M file has 733 tensors. Strata rejected it with `missing qwen35moe.nextn_predict_layers`. The MTP file used above includes 753 tensors. Strata also registers Qwen3.6-35B-A3B under the same profile; omitting `--model-path` selects that separate checkpoint. [Strata model catalog](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md#qwen36-35b-a3b)
