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

Download the 22.85 GB Qwen3.5-35B-A3B MTP GGUF. Strata calls the compatible engine profile `qwen36`. The explicit file path selects the Qwen3.5 weights. This GGUF contains an MTP head. The launch command disables MTP and all other drafting for the common benchmark. [Checkpoint](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-MTP-GGUF)

```sh
model=Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
curl -fL --retry 3 -o "$model" \
  "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-MTP-GGUF/resolve/main/$model"
CUDA_MODULE_LOADING=LAZY STRATA_CUDA=13 vf strata launch --family qwen36 --model UD-Q4_K_XL \
  --model-path "$PWD/$model" --context 8k --parallel 1 --vision off --drafting off --mtp off --tune skip --set STRATA_Q36_BLAS_F32=1 \
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

**MTP and all other drafting were disabled** (`--drafting off --mtp off`). Each of three distinct requests read 227 uncached tokens and generated 64. The replies began with Kyoto, Lyon and Lima. The server reported zero draft proposals and zero acceptances on every request. The decode rate counts the 63 steps after the first token.

| On the same V100-SXM2-32GB, same checkpoint and cuBLAS setting | Median prefill (tokens/s) | Median decode (tokens/s) |
| --- | ---: | ---: |
| Strata CUDA 13 with VF | **1943.5** | **104.55** |
| Native Strata CUDA 12 | 1950.2 | 99.04 |

The same one-slot configuration in an independent pass measured 1940.2 prefill and 104.53 decode. With the default four server slots and one request at a time, VF measured 1940.2 prefill and 103.81 decode. The earlier 152.5 decode measurement used two MTP drafts per step and belongs to the separate speculative experiment. [Non-speculative requests, counters and timings](./benchmarks/strata-no-drafts.json) · [Historical MTP results](./benchmarks/strata-results.json)

## 6. Additional notes

The standard Qwen3.5-35B-A3B Q4_K_M file has 733 tensors. Strata rejected it with `missing qwen35moe.nextn_predict_layers`. The MTP file used above includes 753 tensors. Strata also registers Qwen3.6-35B-A3B under the same profile; omitting `--model-path` selects that separate checkpoint. [Strata model catalog](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md#qwen36-35b-a3b)
