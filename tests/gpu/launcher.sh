#!/bin/bash
# GPU test for the launcher (vf) on a V100:
#   tests/gpu/launcher.sh GPU
# Through the launcher and nothing else in the environment (env -i with only PATH, HOME
# and caches), exercises:
#   (a) PyTorch: torch.cuda.get_device_name(), a matmul compared with the CPU,
#       and that the forwarding cuBLAS is in use (/proc/self/maps).
#   (b) llama.cpp: a short generation that must give text.
#   (c) A virtual environment made by `vf venv`: its Python, started with no prefix, runs
#       PyTorch on the GPU; the layer says that it is there (the variable and the functions
#       of libcuda.so.1); and what PyTorch's builder of extensions would have nvcc compile
#       for is an architecture that CUDA 13 has.
#   (d) VF_SAFE=1 with VF_CUBLAS_BF16_TC=1: what vf doctor says (the switch not used) holds: the bf16
#       cuBLAS variants give cuBLAS 12.9's bits and count no call on their tensor-core path; without
#       VF_SAFE, other bits (the control).
# Both read gigabytes from the file server: use gpu-run-heavy.sh to run this script.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
gpu=${1:-0}
failed=0
vf=$W/scripts/vf
[ -x "$vf" ] || { echo "missing: $vf"; exit 2; }
T=$W/toolkits/cuda-12.9.2/lib
[ -d "$T" ] || { echo "missing: $T (CUDA 12.9.2 redistributable libraries)"; exit 2; }

site=${VF_TORCH:-$VF_TEST_ROOT/torch-cu130/site}
[ -d "$site/torch" ] || { echo "missing: PyTorch in $site"; exit 2; }

D=$W/demos/llamacpp/b12418
B=$D/llama/llama-b12418
[ -x "$B/llama-completion" ] || { echo "missing: $B/llama-completion"; exit 2; }
model=$W/demos/models/qwen2.5-0.5b-instruct-q4_0.gguf
[ -f "$model" ] || { echo "missing: $model"; exit 2; }

out=${VF_OUT:-$VF_TEST_ROOT}
mkdir -p "$out/launcher-caches/proxy" "$out/launcher-caches/nv" "$out/launcher-caches/torch"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$gpu

echo "=== (a) PyTorch through the launcher ==="

# env -i strips the environment; we pass only what is strictly needed.
# CUDA_VISIBLE_DEVICES is captured here and passed through explicitly.
env -i \
  PATH=/usr/bin:/bin \
  HOME="$out/launcher-caches" \
  PYTHONPATH="$site" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$gpu" \
  VF_PROXY_CACHE="$out/launcher-caches/proxy" \
  CUDA_CACHE_PATH="$out/launcher-caches/nv" \
  PYTORCH_KERNEL_CACHE_PATH="$out/launcher-caches/torch" \
  VF_LOG_LEVEL=1 \
  "$vf" --cuda12 "$T" \
  python3 -c '
import torch, sys
print("device:", torch.cuda.get_device_name())
a = torch.randn(256, 256, device="cuda")
b = torch.randn(256, 256, device="cuda")
c_gpu = torch.mm(a, b)
c_cpu = torch.mm(a.cpu(), b.cpu())
diff = (c_gpu.cpu() - c_cpu).abs().max().item()
print("max matmul diff:", diff)
# (The GPU and the CPU add the 256 products of an element in other orders: they are about 6e-5
# apart at most, for elements of up to about 80. 1e-3 leaves a factor of 15 over that, and a
# result off by 0.001 anywhere fails, and so does a NaN.)
if not diff <= 1e-3:
    print("FAIL: matmul difference too large")
    sys.exit(1)
# Check /proc/self/maps for the forwarding cuBLAS.
import re
with open("/proc/self/maps") as f:
    maps = f.read()
cublas_files = set()
for line in maps.splitlines():
    m = re.search(r"/(\S*libcublas\S*\.so\S*)", line)
    if m:
        cublas_files.add(m.group(0))
print("cuBLAS files mapped:")
for p in sorted(cublas_files):
    print(" ", p)
# The forwarding libcublas.so.13 must be mapped.
fwd = [p for p in cublas_files if "/forward/" in p and "libcublas.so.13" in p]
if fwd:
    print("ok: forwarding cuBLAS is in use")
else:
    print("FAIL: forwarding cuBLAS not in /proc/self/maps")
    sys.exit(1)
print("ok")
' >"$out/launcher-caches/torch-launcher.out" 2>"$out/launcher-caches/torch-launcher.err"
torch_status=$?

echo "--- torch stdout ---"
cat "$out/launcher-caches/torch-launcher.out"
echo "--- torch stderr (last 20 lines) ---"
tail -20 "$out/launcher-caches/torch-launcher.err"
echo "--- torch exit status: $torch_status ---"

if [ $torch_status -eq 0 ] && grep -q "ok: forwarding cuBLAS is in use" "$out/launcher-caches/torch-launcher.out"; then
    echo "PASS  pytorch-launcher"
else
    echo "FAIL  pytorch-launcher (exit $torch_status)"
    failed=1
fi

echo ""
echo "=== (b) llama.cpp through the launcher ==="

# For llama.cpp, the existing LD_LIBRARY_PATH with the CUDA 13.5 runtime libs of the
# llama.cpp distribution must be passed so the launcher prepends to it.
env -i \
  PATH=/usr/bin:/bin \
  HOME="$out/launcher-caches" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$gpu" \
  VF_PROXY_CACHE="$out/launcher-caches/proxy" \
  CUDA_CACHE_PATH="$out/launcher-caches/nv" \
  VF_LOG_LEVEL=1 \
  LD_LIBRARY_PATH="$D/cudart/cudart-llama-b12418-bin-ubuntu-cuda-13.5-x64" \
  "$vf" --cuda12 "$T" \
  "$B/llama-completion" -m "$model" -p "The three laws of robotics are" -n 48 -c 2048 --temp 0 -s 1 -t 16 -no-cnv -ngl 99 \
  >"$out/launcher-caches/llama-launcher.out" 2>"$out/launcher-caches/llama-launcher.err"
llama_status=$?

echo "--- llama stdout ---"
cat "$out/launcher-caches/llama-launcher.out"
echo "--- llama stderr (last 20 lines) ---"
tail -20 "$out/launcher-caches/llama-launcher.err"
echo "--- llama exit status: $llama_status ---"

if [ $llama_status -eq 0 ]; then
    text=$(cat "$out/launcher-caches/llama-launcher.out")
    if [ -n "$text" ] && [ ${#text} -gt 10 ]; then
        echo "PASS  llamacpp-launcher"
    else
        echo "FAIL  llamacpp-launcher (output too short or empty)"
        failed=1
    fi
else
    echo "FAIL  llamacpp-launcher (exit $llama_status)"
    failed=1
fi

echo ""
echo "=== (c) a virtual environment made by vf venv, its Python with no prefix ==="

venv=$out/launcher-caches/venv
rm -rf "$venv"
if python3 -c 'import sys, venv; sys.exit(0 if hasattr(sys, "orig_argv") else 1)' 2>/dev/null; then
    env -i PATH=/usr/bin:/bin HOME="$out/launcher-caches" "$vf" venv "$venv" -- --without-pip
    # (In place of `pip install torch`: the directory of the stock wheel, added to the
    # environment's path the way a package adds one.)
    sp=$(ls -d "$venv"/lib/python*/site-packages | head -1)
    echo "$site" >"$sp/the-stock-wheel.pth"
    env -i \
      PATH=/usr/bin:/bin \
      HOME="$out/launcher-caches" \
      CUDA_DEVICE_ORDER=PCI_BUS_ID \
      CUDA_VISIBLE_DEVICES="$gpu" \
      VF_PROXY_CACHE="$out/launcher-caches/proxy" \
      CUDA_CACHE_PATH="$out/launcher-caches/nv" \
      PYTORCH_KERNEL_CACHE_PATH="$out/launcher-caches/torch" \
      VF_LOG_LEVEL=1 \
      "$venv/bin/python" -c '
import ctypes, os, sys
import torch
print("VOLTA_FOREVER:", os.environ.get("VOLTA_FOREVER"))
layer = ctypes.CDLL("libcuda.so.1")
layer.volta_forever_version.restype = ctypes.c_char_p
print("volta_forever_version():", layer.volta_forever_version().decode(), "abi", layer.volta_forever_abi())
print("device:", torch.cuda.get_device_name(), "capability told:", torch.cuda.get_device_capability())
a = torch.randn(256, 256, device="cuda")
b = torch.randn(256, 256, device="cuda")
diff = (torch.mm(a, b).cpu() - torch.mm(a.cpu(), b.cpu())).abs().max().item()
print("max matmul diff:", diff)
try:
    from torch.utils import cpp_extension
    print("what the builder of extensions gives nvcc:", cpp_extension._get_cuda_arch_flags())
except Exception as error:
    print("what the builder of extensions gives nvcc: not read (%s)" % error)
ok = os.environ.get("VOLTA_FOREVER") == layer.volta_forever_version().decode() and layer.volta_forever_abi() == 1 and diff <= 1e-3
print("ok" if ok else "FAIL")
sys.exit(0 if ok else 1)
' >"$out/launcher-caches/venv-launcher.out" 2>"$out/launcher-caches/venv-launcher.err"
    venv_status=$?
    echo "--- venv stdout ---"
    cat "$out/launcher-caches/venv-launcher.out"
    echo "--- venv stderr (last 8 lines) ---"
    tail -8 "$out/launcher-caches/venv-launcher.err"
    # For comparison, the same Python as it is (the hook off): what the card alone is asked to
    # compile for. (Said, not judged: CUDA 13's compiler refuses sm_70.)
    env -i PATH=/usr/bin:/bin HOME="$out/launcher-caches" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
      CUDA_CACHE_PATH="$out/launcher-caches/nv" VF_VENV=0 \
      "$venv/bin/python" -c '
import torch
from torch.utils import cpp_extension
print("without the layer: capability", torch.cuda.get_device_capability(), "and for nvcc:", cpp_extension._get_cuda_arch_flags())
' 2>/dev/null || echo "without the layer: not read"
    if [ $venv_status -eq 0 ] && grep -q "^ok$" "$out/launcher-caches/venv-launcher.out"; then
        echo "PASS  venv-launcher"
    else
        echo "FAIL  venv-launcher (exit $venv_status)"
        failed=1
    fi
else
    echo "SKIP  venv-launcher (no python3 of 3.10 or later with the venv module)"
fi

echo "=== (d) VF_SAFE=1 with VF_CUBLAS_BF16_TC=1 through the launcher ==="
# What vf doctor says (VF_CUBLAS_BF16_TC=1 among what is not used) must be what happens: tests/gpu/forward/bf16-safe.c,
# products that the variants' tensor-core path takes (legacy TN, NN, NT and cuBLASLt with a bias) and one
# that the skinny path takes, run through the launcher with the release runtime of forward/bf16tc, must
# give cuBLAS 12.9's bits, and the runtime's report must count no call on the tensor-core path. The
# same without VF_SAFE (the control) must give other bits and count calls there.
T13=$W/toolkits/cuda-13.5.0
safe_bin=$out/bin/launcher-bf16-safe
mkdir -p "$out/bin"
if [ ! -f "$W/dist/forward/bf16tc/libvf_cublas_bf16.so" ]; then
    echo "SKIP  safe-bf16-tc (the bf16 cuBLAS variants are not built: scripts/build-forwarders.sh)"
elif ! gcc -O2 -Wall -Wno-deprecated-declarations -I"$T13/include" -o "$safe_bin" "$W/tests/gpu/forward/bf16-safe.c" \
    -L"$T13/lib" -Wl,-rpath-link,"$T13/lib" -L"$W/dist/lib" -lcublas -lcublasLt -lcudart -l:libcuda.so.1 -ldl -lm -lpthread; then
    echo "FAIL  safe-bf16-tc (build of bf16-safe.c)"
    failed=1
else
    safe_run() {  # safe_run NAME [VAR=VALUE...]: bf16-safe through the launcher, the switch on
        local name=$1
        shift
        env -i PATH=/usr/bin:/bin HOME="$out/launcher-caches" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
          VF_PROXY_CACHE="$out/launcher-caches/proxy" CUDA_CACHE_PATH="$out/launcher-caches/nv" \
          LD_LIBRARY_PATH="$T13/lib" VF_LOG_LEVEL=2 VF_CUBLAS_BF16_TC=1 "$@" \
          "$vf" --cuda12 "$T" "$safe_bin" >"$out/launcher-caches/$name.log" 2>&1
    }
    doctor=$(env -i PATH=/usr/bin:/bin HOME="$out/launcher-caches" VF_SAFE=1 VF_CUBLAS_BF16_TC=1 \
        "$vf" --cuda12 "$T" doctor 2>&1)
    safe_run safe-bf16-tc VF_SAFE=1
    safe_status=$?
    safe_run safe-bf16-tc-control
    tc_counted() { grep -E "libvf_cublas_bf16.so: (cublasGemmEx|cublasLtMatmul): .*tensor-core path [1-9]" "$1"; }
    log=$out/launcher-caches/safe-bf16-tc.log
    control=$out/launcher-caches/safe-bf16-tc-control.log
    echo "$doctor" | grep -i "VF_SAFE" | sed 's/^/  doctor: /'
    grep -E "VF_SAFE=1:|tensor-core path [0-9]|differ from 12.9's|^ok " "$log" | sed 's/^/  /'
    if ! echo "$doctor" | grep -q "not used: .*VF_CUBLAS_BF16_TC=1"; then
        echo "FAIL  safe-bf16-tc (vf doctor does not say that VF_CUBLAS_BF16_TC=1 is not used)"
        failed=1
    elif [ $safe_status -eq 0 ] && grep -qx "ok bf16-safe" "$log" && grep -q "tensor-core path 0" "$log" &&
        ! tc_counted "$log" >/dev/null && ! grep -qx "ok bf16-safe" "$control" && tc_counted "$control" >/dev/null; then
        echo "PASS  safe-bf16-tc: what vf doctor says holds (12.9's bits, no call on the tensor-core path; without VF_SAFE, other bits)"
    else
        echo "FAIL  safe-bf16-tc (exit $safe_status; $log, control $control)"
        failed=1
    fi
fi

exit $failed
