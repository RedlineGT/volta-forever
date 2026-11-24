#!/bin/bash
# Extract sm_75 cubins from non-PyTorch libraries for the held-out corpus.
# Deduplicates against the PyTorch cubins in $TORCH.
#
#   tests/sass/heldout-extract.sh OUTDIR
#
# Sources: CUDA 13.5 toolkit, cuDNN, cuSPARSELt, NCCL, nvshmem (from the PyTorch wheel),
# cu13 libraries (from the PyTorch wheel), and vLLM compiled extensions. Each directory is set
# by a variable; the defaults are under $VF_TEST_ROOT.
set -eu
W=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$W/tests/site.env" ] && . "$W/tests/site.env"
: "${VF_TEST_ROOT:=${XDG_CACHE_HOME:-$HOME/.cache}/volta-forever/tests}"
# (The tool: as tests/sass/compare.sh finds it.)
if [ -z "${VF_TOOL:-}" ]; then
    if [ -n "${CARGO_TARGET_DIR:-}" ]; then VF_TOOL=$CARGO_TARGET_DIR/release/vf-translate
    elif [ -x "$W/target/release/vf-translate" ]; then VF_TOOL=$W/target/release/vf-translate
    else VF_TOOL=$(. "$W/scripts/target-dir.sh" && vf_target_dir "$W")/release/vf-translate
    fi
fi
TORCH=${TORCH:-$VF_TEST_ROOT/torch-cubins}
[ -x "$VF_TOOL" ] || { echo "no tool at $VF_TOOL" >&2; exit 2; }

OUTDIR=${1:?usage: heldout-extract.sh OUTDIR}
mkdir -p "$OUTDIR"

# Build PyTorch hash index
torch_hashes=$(mktemp)
trap 'rm -f "$torch_hashes"' EXIT
for f in "$TORCH"/*.cubin; do sha256sum "$f"; done | awk '{print $1}' | sort -u > "$torch_hashes"

extract_lib() {
    local lib=$1 name=$2 dir=$OUTDIR/$name
    [ -f "$lib" ] || return 0
    mkdir -p "$dir"
    "$VF_TOOL" extract "$lib" "$dir" 75 2>&1 || true
    local total=$(ls "$dir"/*.sm_75.cubin 2>/dev/null | wc -l)
    [ "$total" -gt 0 ] || { rm -rf "$dir"; return 0; }
    local dupes=0
    for f in "$dir"/*.sm_75.cubin; do
        h=$(sha256sum "$f" | awk '{print $1}')
        grep -q "^$h$" "$torch_hashes" && { rm "$f"; dupes=$((dupes+1)); }
    done
    local left=$(ls "$dir"/*.sm_75.cubin 2>/dev/null | wc -l)
    echo "$name: $total extracted, $dupes PyTorch duplicates, $left kept"
    [ "$left" -gt 0 ] || rm -rf "$dir"
}

CUDA135=${VF_HELDOUT_CUDA135:-$W/toolkits/cuda-13.5.0/lib}
CUDNN=${VF_HELDOUT_CUDNN:-$VF_TEST_ROOT/torch-cu130/site/nvidia/cudnn/lib}
CU13=${VF_HELDOUT_CU13:-$VF_TEST_ROOT/torch-cu130/site/nvidia/cu13/lib}
CUSPARSELT=${VF_HELDOUT_CUSPARSELT:-$VF_TEST_ROOT/torch-cu130/site/nvidia/cusparselt/lib}
NCCL=${VF_HELDOUT_NCCL:-$VF_TEST_ROOT/torch-cu130/site/nvidia/nccl/lib}
NVSHMEM=${VF_HELDOUT_NVSHMEM:-$VF_TEST_ROOT/torch-cu130/site/nvidia/nvshmem/lib}
VLLM=${VF_HELDOUT_VLLM:-$VF_TEST_ROOT/vllm/site}

for lib in "$CUDA135"/lib*.so.*; do [ -L "$lib" ] && continue; name=$(basename "$lib" | sed 's/\.so\..*//'); extract_lib "$lib" "cuda135-$name"; done
for lib in "$CUDNN"/lib*.so.*; do [ -L "$lib" ] && continue; name=$(basename "$lib" | sed 's/\.so\..*//'); extract_lib "$lib" "cudnn-$name"; done
for lib in "$CU13"/lib*.so.*; do [ -L "$lib" ] && continue; name=$(basename "$lib" | sed 's/\.so\..*//'); extract_lib "$lib" "cu13-$name"; done
for lib in "$CUSPARSELT"/lib*.so.*; do [ -L "$lib" ] && continue; name=$(basename "$lib" | sed 's/\.so\..*//'); extract_lib "$lib" "cusparselt-$name"; done
for lib in "$NCCL"/lib*.so.*; do [ -L "$lib" ] && continue; name=$(basename "$lib" | sed 's/\.so\..*//'); extract_lib "$lib" "nccl-$name"; done
for lib in "$NVSHMEM"/lib*.so.*; do [ -L "$lib" ] && continue; name=$(basename "$lib" | sed 's/\.so\..*//'); extract_lib "$lib" "nvshmem-$name"; done
for lib in "$VLLM"/vllm/*.so "$VLLM"/vllm/vllm_flash_attn/*.so "$VLLM"/vllm/third_party/deep_gemm/*.so; do
    [ -f "$lib" ] && [ ! -L "$lib" ] || continue
    name=$(basename "$lib" .so | sed 's/\.cpython[^.]*//; s/\.abi3//')
    extract_lib "$lib" "vllm-$name"
done

echo ""; echo "Total: $(find "$OUTDIR" -name '*.sm_75.cubin' | wc -l) cubins in $(du -sh "$OUTDIR" | awk '{print $1}')"
