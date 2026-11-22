# llama.cpp on Ubuntu with a Tesla V100

## 1. CUDA

On Ubuntu 24.04 x86-64, check for the NVIDIA R580 driver. Reboot if you install it.

```bash
if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | grep -E 'V100.*, 580[.]'; then
  sudo apt update && sudo apt install -y nvidia-driver-580
fi
```

The official CUDA 13 server ships with a separate CUDA runtime archive. Only the native comparison build needs the CUDA 12.9 toolkit. Install it if the compiler is missing. [NVIDIA CUDA packages for Ubuntu 24.04](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/)

```bash
if ! /usr/local/cuda-12.9/bin/nvcc --version 2>/dev/null | grep -q 'release 12.9'; then
  sudo apt update && sudo apt install -y ca-certificates curl
  repo=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64
  curl -fsSL "$repo/cuda-archive-keyring.gpg" | sudo tee /usr/share/keyrings/cuda-archive-keyring.gpg >/dev/null
  echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] $repo/ /" | sudo tee /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list
  sudo apt update && sudo apt install -y cuda-toolkit-12-9
fi
```

## 2. VF

```bash
sudo apt update && sudo apt install -y curl python3 xz-utils
curl -fsSL https://raw.githubusercontent.com/nbald/volta-forever/main/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
vf fetch --all
```

## 3. Engine

Download the CUDA 13.5 server and runtime for the release selected by upstream. Build the native V100 server from that same tag. [Official releases](https://github.com/ggml-org/llama.cpp/releases) · [CUDA build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cuda)

```bash
(
set -euo pipefail
sudo apt install -y git build-essential cmake ninja-build libssl-dev
mkdir -p ~/llama-v100/cuda13 && cd ~/llama-v100
releases=https://github.com/ggml-org/llama.cpp/releases
tag=$(curl -fsSL "$releases/latest/download/nightly-tag.txt")
for name in llama cudart-llama; do
  curl -fL --retry 3 "$releases/download/$tag/$name-$tag-bin-ubuntu-cuda-13.5-x64.tar.gz" |
    tar -xz --strip-components=1 -C cuda13
done
git clone --depth 1 --branch "$tag" https://github.com/ggml-org/llama.cpp.git src
cmake -S src -B native -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_OPENSSL=OFF
cmake --build native --target llama-server --parallel 2
)
```

## 4. Test

Download the GGUF once. Both servers use this exact file. [Unsloth Qwen3.5-35B-A3B UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-MTP-GGUF/blob/63af8373893a7a73c6dfcb84cb63d815981da5e0/Qwen3.5-35B-A3B-UD-Q4_K_M.gguf)

```bash
cd ~/llama-v100
model=Qwen3.5-35B-A3B-UD-Q4_K_M.gguf
curl -fL --retry 3 -C - -o "$model" "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-MTP-GGUF/resolve/63af8373893a7a73c6dfcb84cb63d815981da5e0/$model" &&
echo "a20c5a9f0cee94d24af3553e7828df1337c7d24213789911c74e1f4936894936  $model" | sha256sum -c - &&
args=(-m "$model" --alias qwen --host 127.0.0.1 --port 8080 -ngl 999 -np 1 -c 512 -fa on --cache-ram 0 --spec-type none --jinja) &&
LD_LIBRARY_PATH="$PWD/cuda13${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" vf ./cuda13/llama-server "${args[@]}"
```

Wait for the server to listen, then send a request from a second terminal. The expected visible reply is **Paris**.

```bash
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"What is the capital of France? Answer with just the city name."}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}' |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
```

For the native comparison, stop the CUDA 13 server with Ctrl-C and run this in its terminal. Repeat the same request.

```bash
LD_LIBRARY_PATH="$PWD/native/bin:/usr/local/cuda-12.9/lib64" ./native/bin/llama-server "${args[@]}"
```

## 5. Benchmarks

Run the client once per server, with one GPU and one server running. Its four requests use an independent warm-up, then Kyoto, Lyon and Lima with up to 64 output tokens. It disables prompt reuse and reads the prefill and decode timers returned by llama.cpp. [Common prompts](./benchmarks/prompts.json)

```bash
cd ~/llama-v100
base=https://raw.githubusercontent.com/nbald/volta-forever/main/docs/guides/benchmarks
curl -fsSLO "$base/freetoken.py" && curl -fsSLO "$base/prompts.json"
python3 freetoken.py --url http://127.0.0.1:8080 --disable-prompt-cache
```

Both b12418 servers answered **Paris**. After warm-up, each processed three 227-token prompts with zero cached tokens and generated 64 tokens per prompt. Each server ran in a fresh Ubuntu 24.04.5 environment. The table uses the server’s own prefill and decode timers. [Per-request timings and replies](./benchmarks/common-gguf-q4-k-m.json)

| Build on the same V100 | Median prefill (tokens/s) | Median decode (tokens/s) |
| --- | ---: | ---: |
| Native sm_70, CUDA 12.9.2 | 494.57 | 101.27 |
| Official CUDA 13.5 archive with VF 0.1.1 | 1089.57 | 101.11 |

## 6. Additional notes

Both commands offload all model layers to the V100, enable flash attention and run one sequence at a time. MTP and speculative decoding are disabled. The CUDA 12.9 build runs directly on sm_70. The CUDA 13 build uses VF.
