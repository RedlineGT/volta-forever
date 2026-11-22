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

Build VF from the main branch for the benchmark below. The build fetches its CUDA libraries into the checkout.

```sh
sudo apt update && sudo apt install -y build-essential ca-certificates curl git python3 xz-utils
if ! command -v rustup >/dev/null; then
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"
git clone --depth 1 https://github.com/nbald/volta-forever.git ~/volta-forever
(cd ~/volta-forever && scripts/build.sh && scripts/fetch-all.sh redist && scripts/build-forwarders.sh)
export PATH="$HOME/volta-forever/scripts:$PATH"
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
  "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-MTP-GGUF/resolve/63af8373893a7a73c6dfcb84cb63d815981da5e0/$model"
CUDA_MODULE_LOADING=LAZY STRATA_CUDA=13 vf strata launch --family qwen36 --model UD-Q4_K_XL \
  --model-path "$PWD/$model" --context 8k --parallel 1 --vision off --drafting off --mtp off --tune skip --set STRATA_Q36_BLAS_F32=1 \
  --set prompt-cache=0 --set conversation-cache-mib=0 --access local --port 9085 \
  --no-api-key --yes --no-tui
```

Ask “What is the capital of France? Answer in one word.” in the local chat. The unpinned installation selected the signed Strata 0.2.18 terminal and CUDA 13 engine 0.1.64. It answered **Paris** on a Tesla V100-SXM2-32GB with driver 580.178.04 with both VF 0.1.1 and the VF build used below.

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
| Strata CUDA 13 with VF from main | **1982.5** | **104.25** |
| Native Strata CUDA 12 | 1948.5 | 99.07 |

A fresh Ubuntu 24.04 build of VF commit `85509eb` served the same three requests at medians of **1980.80 tokens/s prefill** and **104.22 tokens/s decode**. [Build and request records](./benchmarks/strata-uniform-divisor.json)

The performance targets set before this run were 1950.2 prefill and 99.04 decode tokens/s. To repeat the native comparison, stop the VF server, select the CUDA 12 engine, and repeat section 4 with `STRATA_CUDA=12 strata launch`. Use the same model file and launch options.

```sh
strata launch --stop --no-tui --yes
STRATA_CUDA=12 strata update --backend cuda --no-control
```

The earlier VF release measured 1943.5 prefill and 104.55 decode. The newer VF build optimizes repeated integer divisions in the GPU kernel that groups tokens by expert. [Measured requests and binaries](./benchmarks/strata-uniform-divisor.json) · [VF source](https://github.com/nbald/volta-forever/commit/ff059a358a9104d3bd007b1336260592f66bfd0e) · [Earlier release measurements](./benchmarks/strata-no-drafts.json)

## 6. Additional notes

The standard Qwen3.5-35B-A3B Q4_K_M file has 733 tensors. Strata rejected it with `missing qwen35moe.nextn_predict_layers`. The MTP file used above includes 753 tensors. Strata also registers Qwen3.6-35B-A3B under the same profile; omitting `--model-path` selects that separate checkpoint. [Strata model catalog](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md#qwen36-35b-a3b)
