# volta-forever

Run CUDA 13 applications on a Tesla V100. Prefix a command with `vf` to run supported GPU code on Volta using NVIDIA’s R580 driver. This works with stock PyTorch wheels and the official CUDA 13 build of llama.cpp.

[Inference engines and Ubuntu guides](#inference-engines) · [Installation](INSTALL.md) · [Compatibility limits](LIMITS.md)

## Quick start

You need a Tesla V100, the NVIDIA R580 driver, Linux x86-64 with glibc 2.34 or later, and `curl`, `python3` and `xz`. Ubuntu 22.04+ and Debian 12+ meet the glibc requirement. The installer places `vf` in `~/.local/bin` and downloads NVIDIA’s CUDA 12.9.2 cuBLAS.

```sh
curl -fsSL https://raw.githubusercontent.com/nbald/volta-forever/main/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
vf doctor
```

Run an existing CUDA program, or use `vf venv` to install ordinary CUDA wheels.

```sh
vf llama-completion -m model.gguf -p "Hello" -n 48 -ngl 99

vf venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -c "import torch; print(torch.cuda.get_device_name())"
```

Run `vf fetch --all` if the application needs more CUDA 12.9 libraries. Builds from `main` also fetch NVIDIA’s CUDA 12 NCCL for multi-GPU collectives. [Installation and launcher options](INSTALL.md)

## How it works

CUDA 13 applications ship kernels for GPUs newer than Volta. VF intercepts module loads, retargets supported PTX and translates supported Turing (`sm_75`) machine code to Volta (`sm_70`). The NVIDIA driver runs the resulting kernels on the V100. VF caches translations and routes libraries such as cuBLAS to their CUDA 12.9 implementations.

CUDA 13 programs see capability 7.5 by default. Applications with native Volta kernels keep 7.0. To select an application’s Ampere kernels, run `VF_CAPABILITY=8.0 vf ./my_program`. VF can emulate instructions used by BF16, FP8 and NVFP4 kernels. Builds from `main` also handle selected `sm_80` machine code and PTX retained inside a cubin. When a kernel needs an unsupported instruction, VF reports the kernel name and the reason. [Architecture](docs/architecture.md) · [Instruction emulation](docs/emulations.md) · [Supported images and limits](LIMITS.md)

## Inference engines

These benchmarks use **Qwen3.5-35B-A3B on one Tesla V100-SXM2-32GB**, with **MTP and all other speculative decoding disabled**. Each server handled one request at a time. After a warm-up, it processed three distinct 227-token prompts and generated 64 tokens per prompt. The table reports median **prefill / decode** in tokens/s.

The native references are 1Cat-vLLM with CUDA 12.8, upstream Strata with CUDA 12.9 and llama.cpp with CUDA 12.9.2. Each native/VF pair used the same physical card and the same checkpoint. The Ubuntu 24.04 guides include installation, a model test and benchmark commands.

| Ubuntu guide | 35B checkpoint format | With VF, prefill / decode | Native, prefill / decode |
|---|---|---:|---:|
| [vLLM](docs/guides/vllm.md) | QuantTrio AWQ W4A16 | 1,449.98 / 80.23 | 1,443.33 / 76.09 |
| [SGLang](docs/guides/sglang.md) | QuantTrio AWQ W4A16 | 2,016.06 / 77.33 | — |
| [llama.cpp](docs/guides/llamacpp.md) | Unsloth GGUF Q4_K_M | 1,076.51 / 99.12 | 497.33 / 100.43 |
| [FreeToken](docs/guides/freetoken.md) | Unsloth GGUF Q4_K_M | 884.56 / 109.26 | — |
| [Strata](docs/guides/strata.md) | Unsloth GGUF UD-Q4_K_XL with MTP weights | 1,982.50 / 104.25 | 1,948.50 / 99.07 |
| [ZML / LLMD](docs/guides/zml.md) | AMD MXFP4 | 681.68¹ / 46.19 | — |
| [NInfer](docs/guides/ninfer.md) | Mixed Q4/Q5/Q6/Q8 `.ninfer` | 556.51 / 117.08 | — |
| [TensorFold](docs/guides/tensorfold.md) | MLX 4-bit experts, 8-bit routers | 464.25 / 57.67 | — |

¹ LLMD’s prefill figure uses GPU timestamps for the complete prefill graph. Measured through the server’s first token, the rate was 671.60 input tokens/s. Decode uses LLMD’s own step timer. [Phase timings](docs/guides/benchmarks/zml-llmd-autotune.json)

llama.cpp and FreeToken share the same Q4_K_M file. Strata requires MTP metadata at load time even with drafting off. Its guide identifies the exact file used. [Checkpoint revisions, per-request timings and replies](docs/guides/benchmarks/README.md)

## Validation

On V100 with driver 580.178.04, PyTorch matched the CPU on all 924 operations with a CPU reference and all 32 small models. vLLM and SGLang matched the CPU reference token for token on greedy Qwen2.5-0.5B tests. Selected low-precision operations were also checked against an RTX 5090. Coverage depends on the kernel. [Validation results](docs/validation.md) · [Performance measurements](docs/performance.md) · [Known limits](LIMITS.md)

## Build and test

To build from source, install Rust (edition 2024) and GCC, then run these commands from the checkout.

```sh
scripts/build.sh
scripts/fetch-all.sh redist
scripts/build-forwarders.sh
scripts/vf doctor
cargo test --workspace
```

Run `tests/gpu/run.sh` for the physical V100 suite. Use `scripts/package.sh` to create a release archive. [Test suites](tests/README.md) · [Contributing](CONTRIBUTING.md) · [Machine-code translator](docs/translator.md)

## Licence

Apache License 2.0. The instruction tables derived from denvdis are licensed under LGPL 2.1. NVIDIA and CUDA are trademarks of NVIDIA Corporation. This project is independent of NVIDIA. [License](LICENSE) · [Instruction table licence](crates/vf-tables/LICENSE) · [Notices](NOTICE)
