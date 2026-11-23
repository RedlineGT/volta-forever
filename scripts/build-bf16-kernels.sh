#!/bin/bash
# Build the kernels of the bf16 cuBLAS runtime: crates/vf-fwdgen/bf16tc/kernels.cu into crates/vf-fwdgen/bf16tc/kernels.fatbin,
# sm_70 machine code only, with the CUDA 12.9.2 compiler of toolkits/, and write
# crates/vf-fwdgen/bf16tc/kernels.manifest (the compiler, the flags, the hashes of the sources and of the result).
# Both files are checked in: the forwarding libraries are built from them without a CUDA compiler.
#   scripts/build-bf16-kernels.sh            rebuild both files
#   scripts/build-bf16-kernels.sh --check    rebuild in a temporary directory and compare with the
#                                            checked-in files (exit status 1 if they differ)
# Refuses a result that holds PTX, or whose intermediate PTX calls a function from outside the
# source (`.extern .func`: a library routine such as libdevice's).
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
src=$root/crates/vf-fwdgen/bf16tc
nvcc=${NVCC:-$root/toolkits/cuda-12.9.2/bin/nvcc}
cuobjdump=$(dirname "$nvcc")/cuobjdump
check=0
[ "${1:-}" = --check ] && check=1
[ -x "$nvcc" ] || { echo "build-bf16-kernels: no CUDA 12.9.2 compiler at $nvcc" >&2; exit 2; }
version=$("$nvcc" --version | tail -1)
case $version in *cuda_12.9.r12.9*) ;; *) echo "build-bf16-kernels: $nvcc is not CUDA 12.9 ($version)" >&2; exit 2 ;; esac
flags=(-gencode arch=compute_70,code=sm_70 -Wno-deprecated-gpu-targets -O3 -fmad=false --compress-mode=none -frandom-seed=vf-bf16 -std=c++17)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp "$src/kernels.cu" "$src/abi.h" "$work/"
# The source includes nothing but its own header (nvcc adds cuda_runtime.h by itself).
if grep -n '^[[:space:]]*#[[:space:]]*include' "$src/kernels.cu" "$src/abi.h" | grep -v '"abi.h"'; then
    echo "build-bf16-kernels: the kernels include a header other than abi.h" >&2
    exit 1
fi
# Relative names in a directory of its own: nothing of the checkout's path reaches the output.
(cd "$work" && "$nvcc" "${flags[@]}" -fatbin -o kernels.fatbin kernels.cu)
(cd "$work" && "$nvcc" -arch=compute_70 -Wno-deprecated-gpu-targets -O3 -fmad=false -frandom-seed=vf-bf16 -std=c++17 -ptx -o kernels.ptx kernels.cu)
if grep -q '\.extern \.func' "$work/kernels.ptx"; then
    echo "build-bf16-kernels: the PTX calls functions from outside the source:" >&2
    grep '\.extern \.func' "$work/kernels.ptx" >&2
    exit 1
fi
inspect() {  # inspect FILE: sm_70 machine code only, one image, no PTX
    if [ -n "$("$cuobjdump" -lptx "$1" 2>/dev/null)" ]; then
        echo "build-bf16-kernels: $1 holds PTX" >&2
        return 1
    fi
    local elfs
    elfs=$("$cuobjdump" -lelf "$1")
    if [ "$(echo "$elfs" | grep -c 'sm_70')" != 1 ] || [ "$(echo "$elfs" | wc -l)" != 1 ]; then
        echo "build-bf16-kernels: $1 does not hold exactly one sm_70 image: $elfs" >&2
        return 1
    fi
}
inspect "$work/kernels.fatbin"
hash() { sha256sum "$1" | cut -d' ' -f1; }
{
    echo "# What scripts/build-bf16-kernels.sh made kernels.fatbin from. Checked by the script (--check), by"
    echo "# vf-fwdgen's unit tests and verify, and by scripts/package.sh (vf-fwdgen check-bf16)."
    echo "compiler $version"
    echo "flags ${flags[*]}"
    echo "source kernels.cu $(hash "$work/kernels.cu")"
    echo "source abi.h $(hash "$work/abi.h")"
    echo "fatbin kernels.fatbin $(hash "$work/kernels.fatbin")"
} > "$work/kernels.manifest"
if [ "$check" = 1 ]; then
    bad=0
    inspect "$src/kernels.fatbin" || bad=1
    cmp -s "$work/kernels.fatbin" "$src/kernels.fatbin" || { echo "FAIL  the rebuilt kernels.fatbin differs from the checked-in one"; bad=1; }
    cmp -s "$work/kernels.manifest" "$src/kernels.manifest" || { echo "FAIL  the manifest differs:"; diff "$src/kernels.manifest" "$work/kernels.manifest" || true; bad=1; }
    [ "$bad" = 0 ] && echo "PASS  kernels.fatbin and kernels.manifest are what the source and the compiler give ($(stat -c %s "$src/kernels.fatbin") bytes, sm_70 machine code only, no external function)"
    exit "$bad"
fi
cp "$work/kernels.fatbin" "$work/kernels.manifest" "$src/"
echo "built $src/kernels.fatbin ($(stat -c %s "$src/kernels.fatbin") bytes) and kernels.manifest"
