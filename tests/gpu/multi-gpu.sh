#!/usr/bin/env bash
# One-process multi-GPU concurrency test: concurrent module/library load, concurrent graph
# capture/replay, and cross-device cuMemcpyPeerAsync pipeline against native sm_70 cubin, VF
# retargeted PTX, and VF translated cubin under both LAZY and EAGER module loading modes.
set -eu

W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
vf_clear_settings

PROXY=${VF_PROXY:-$W/dist/lib}
OUT=${VF_OUT:-$VF_TEST_ROOT}/multi-gpu
BIN=$OUT/multi-gpu
CACHE_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/vf-multi-gpu-cache.XXXXXX")
trap 'rm -rf -- "$CACHE_ROOT"' EXIT

mkdir -p "$OUT" "$OUT/cuda-cache"
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$OUT/cuda-cache}
${CC:-gcc} -std=gnu11 -O2 -Wall -Wextra -Werror "$W/tests/gpu/multi-gpu.c" -pthread -ldl -o "$BIN"

run_case() {
    name=$1
    shift
    if "$@" >/dev/null 2>"$OUT/$name.err"; then
        echo "PASS  $name"
    else
        rc=$?
        if [ "$rc" -eq 77 ]; then
            echo "SKIP  $name (< 2 GPUs visible)"
            return 77
        fi
        echo "FAIL  $name (exit $rc)"
        cat "$OUT/$name.err" >&2 || true
        exit 1
    fi
}

rc=0
run_case native-sm70 "$BIN" "$W/tests/kernels/k_basic.128.sm70.cubin" || rc=$?
if [ "$rc" -eq 77 ]; then
    exit 2
fi

rm -rf "$CACHE_ROOT/ptx-lazy" "$CACHE_ROOT/ptx-eager" "$CACHE_ROOT/cubin-lazy" "$CACHE_ROOT/cubin-eager"
run_case vf-ptx-lazy \
    env LD_LIBRARY_PATH="$PROXY${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        CUDA_MODULE_LOADING=LAZY \
        VF_PROXY_CACHE="$CACHE_ROOT/ptx-lazy" \
        "$BIN" "$W/tests/kernels/k_basic.135.sm75.ptx"

run_case vf-ptx-eager \
    env LD_LIBRARY_PATH="$PROXY${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        CUDA_MODULE_LOADING=EAGER \
        VF_PROXY_CACHE="$CACHE_ROOT/ptx-eager" \
        "$BIN" "$W/tests/kernels/k_basic.135.sm75.ptx"

run_case vf-cubin-lazy \
    env LD_LIBRARY_PATH="$PROXY${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        CUDA_MODULE_LOADING=LAZY \
        VF_PROXY_CACHE="$CACHE_ROOT/cubin-lazy" \
        "$BIN" "$W/tests/kernels/k_basic.135.sm75.cubin"

run_case vf-cubin-eager \
    env LD_LIBRARY_PATH="$PROXY${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        CUDA_MODULE_LOADING=EAGER \
        VF_PROXY_CACHE="$CACHE_ROOT/cubin-eager" \
        "$BIN" "$W/tests/kernels/k_basic.135.sm75.cubin"

run_case vf-ptx-nocache \
    env LD_LIBRARY_PATH="$PROXY${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        VF_PROXY_CACHE=0 \
        "$BIN" "$W/tests/kernels/k_basic.135.sm75.ptx"

run_case vf-cubin-nocache \
    env LD_LIBRARY_PATH="$PROXY${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        VF_PROXY_CACHE=0 \
        "$BIN" "$W/tests/kernels/k_basic.135.sm75.cubin"
