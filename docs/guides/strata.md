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

Build VF 0.1.3 for the benchmark below. The build fetches its CUDA libraries into the checkout.

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

Download the 22.63 GB UD-Q4_K_M GGUF. Strata uses the catalog profile `qwen36` and the size preset `UD-Q4_K_XL`. The explicit `--model-path` selects the verified Qwen3.5 UD-Q4_K_M file. [Checkpoint](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-MTP-GGUF)

```sh
model=Qwen3.5-35B-A3B-UD-Q4_K_M.gguf
curl -fL --retry 3 -o "$model" \
  "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-MTP-GGUF/resolve/63af8373893a7a73c6dfcb84cb63d815981da5e0/$model" &&
printf '%s  %s\n' a20c5a9f0cee94d24af3553e7828df1337c7d24213789911c74e1f4936894936 "$model" | sha256sum --check &&
CUDA_MODULE_LOADING=LAZY STRATA_CUDA=13 vf strata launch --family qwen36 --model UD-Q4_K_XL \
  --model-path "$PWD/$model" --context 8k --parallel 1 --vision off --drafting off --mtp off --tune skip --set STRATA_Q36_BLAS_F32=1 \
  --set prompt-cache=0 --set conversation-cache-mib=0 --access local --port 9085 \
  --no-api-key --yes --no-tui
```

Ask “What is the capital of France? Answer in one word.” in the local chat. The unpinned installation selected the signed Strata 0.2.18 terminal and CUDA 13 engine 0.1.64. It answered **Paris** on a Tesla V100-SXM2-32GB with driver 580.178.04 with the CUDA 12 native engine and with CUDA 13 under VF.

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
| Strata CUDA 13 with VF from main | **2000.00** | **109.13** |
| Native Strata CUDA 12.9 | 1968.80 | 105.19 |

Both engines passed the same four prompts on the same V100. [Per-request timings and replies](./benchmarks/common-gguf-q4-k-m.json)

To repeat the native comparison, stop the VF server, select the CUDA 12 engine, and repeat section 4 with `STRATA_CUDA=12 strata launch`. Keep `--model-path` pointed at the GGUF verified in section 4 and keep all other launch options the same.

```sh
strata launch --stop --no-tui --yes
STRATA_CUDA=12 strata update --backend cuda --no-control
```

## 6. Additional notes

The standard Q4_K_M from the separate Unsloth repository lacks `qwen35moe.nextn_predict_layers`. Both Strata builds required this metadata even with MTP disabled. The UD-Q4_K_M file used here also runs in llama.cpp and FreeToken. [Common checkpoint and measurements](./benchmarks/common-gguf-q4-k-m.json)
