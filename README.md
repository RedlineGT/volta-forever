# volta-forever

**Your Tesla V100 is not dead.**\
CUDA 13 dropped Volta.\
volta-forever brings it back.

Run unmodified CUDA 13 applications on a Tesla V100 with a simple two-letter command prefix: `vf`.

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

All eight engines ran **Qwen3.5-35B-A3B** on a Tesla V100-SXM2-32GB with **MTP and all other speculative decoding disabled**. Each engine processed one warm-up followed by three distinct 227-token prompts at concurrency 1 and generated 64 tokens per prompt. Rates are medians in tokens/s. Each native/VF pair used the same card and checkpoint.

| Engine (Ubuntu guide) | Tested model / quantization | CUDA 12 prefill | CUDA 12 decode | CUDA 13 + VF prefill | CUDA 13 + VF decode |
|---|---|---:|---:|---:|---:|
| [vLLM](docs/guides/vllm.md) | QuantTrio AWQ W4A16 | 1,443.33¹ | 76.09¹ | 1,449.98 | 80.23 |
| [SGLang](docs/guides/sglang.md) | QuantTrio AWQ W4A16 | Can't run | Can't run | 2,016.06 | 77.33 |
| [llama.cpp](docs/guides/llamacpp.md) | Unsloth GGUF Q4_K_M | 497.33 | 100.43 | 1,076.51 | 99.12 |
| [FreeToken](docs/guides/freetoken.md) | Unsloth GGUF Q4_K_M | Can't run | Can't run | 884.56 | 109.26 |
| [Strata](docs/guides/strata.md) | Unsloth GGUF UD-Q4_K_XL | 1,948.50 | 99.07 | 1,982.50 | 104.25 |
| [ZML / LLMD](docs/guides/zml.md) | AMD MXFP4 | Can't run | Can't run | 681.68² | 46.19 |
| [NInfer](docs/guides/ninfer.md) | Mixed Q4/Q5/Q6/Q8 `.ninfer` | Can't run | Can't run | 556.51 | 117.08 |
| [TensorFold](docs/guides/tensorfold.md) | MLX 4-bit experts, 8-bit routers | Can't run | Can't run | 464.25 | 57.67 |

¹ The CUDA 12 figures for vLLM are from **1Cat-vLLM** running CUDA 12.8. The native Strata and llama.cpp builds used CUDA 12.9 and 12.9.2 respectively. [1Cat-vLLM measurements](docs/guides/benchmarks/vllm-graphs.json)

² LLMD prefill uses the GPU span of its complete prefill graph. The server measured 671.60 input tokens/s through the first token. Decode uses its own step timer. [Phase timings](docs/guides/benchmarks/zml-llmd-autotune.json)

Compatibility applies to the listed checkpoint and upstream engine on a native V100. [Checkpoint revisions and per-request results](docs/guides/benchmarks/README.md)

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
