#!/bin/bash
# vLLM's top-k and top-p sampling kernel (_topk_topp_kernel, Triton) against the same kernel run on
# the CPU by Triton's interpreter (tests/gpu/vllm-topk-topp-test.py, 18 cases), in two passes:
#   1. VF_BORROW=1: the translator tries levels 2, 1, then 0 for every kernel, and refuses one
#      that needs more (docs/translator.md, "What VF_BORROW chooses");
#   2. VF_BORROW=5, with a proxy cache of its own: level 3 with room made where a kernel needs it,
#      the translation that the second translation at launch makes (the setting with which the
#      kernels refused in the first pass were translated for the second until 2026-11-23).
# (Until 2026-11-23 the second pass ran at VF_BORROW=1 as well: it wrote translations made by
# reference/translate.py where a hook of the proxy, VF_DEBUG_TRANSLATED, would have taken them, and
# that hook was gone before 0.1.0. Each pass has its own proxy cache now, so that the second
# cannot take what the first translated.)
#
# Run it on a machine with a V100, the GPU chosen by CUDA_VISIBLE_DEVICES. It needs $VF_PROXY,
# Python 3.12 headers (for Triton's C stub) and the vLLM and PyTorch sites. The exit status is 1
# when a case of either pass fails.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
out=${VF_OUT:-$VF_TEST_ROOT}

proxy=${VF_PROXY:-$W/dist/lib}
vllm_site=${VF_VLLM_SITE:-$VF_TEST_ROOT/vllm/site}
torch_site=${VF_TORCH:-$VF_TEST_ROOT/torch-cu130/site}
extra=${VF_VLLM_EXTRA:-$VF_TEST_ROOT/vllm-extra/site}
pydev=${VF_PYTHON_DEV:-$VF_TEST_ROOT/python-dev}
map=${VF_AUDIT_MAP_FILE:-$out/torch-audit.map}

cache=$out/vllm-topk-cache
mkdir -p "$cache/tmp" "$cache/.cache" "$cache/triton" "$cache/hf"

export HOME=$cache TMPDIR=$cache/tmp XDG_CACHE_HOME=$cache/.cache
export TRITON_CACHE_DIR=$cache/triton HF_HOME=$cache/hf
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# (A caller's VF_PROXY_CACHE=0, no cache, holds for both passes.)
no_cache=${VF_PROXY_CACHE:-}
. "$W/tests/gpu/cache.sh"
export PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1

pp="$vllm_site:$torch_site"
[ -d "$extra" ] && pp="$extra:$pp"
export PYTHONPATH="$pp"
export LD_LIBRARY_PATH="$proxy"
[ -f "$map" ] && export LD_AUDIT="$proxy/libvf_audit.so" VF_AUDIT_MAP_FILE="$map"
[ -d "$pydev/usr/include/python3.12" ] && export CPATH="$pydev/usr/include/python3.12:$pydev/usr/include"
export LIBRARY_PATH="$proxy"
export VF_LOG_LEVEL=1

# Each pass with a proxy cache of its own, under /tmp as tests/gpu/cache.sh makes one (not on the
# NFS, where the proxy refuses root's files), removed at the end with the one cache.sh made; or
# none, when the caller said VF_PROXY_CACHE=0.
caches=()
trap 'vf_cleanup_proxy_cache; rm -rf -- "${caches[@]}"' EXIT
status=0
for borrow in 1 5; do
    echo ""
    echo "=== Pass with VF_BORROW=$borrow ==="
    proxy_cache=0
    if [ "$no_cache" != 0 ]; then
        proxy_cache=$(mktemp -d /tmp/vf-cache-XXXXXXXX) || exit 1
        caches+=("$proxy_cache")
    fi
    VF_BORROW=$borrow VF_PROXY_CACHE=$proxy_cache python3 -s "$W/tests/gpu/vllm-topk-topp-test.py" 2>&1
    pass=$?
    echo "=== VF_BORROW=$borrow: status $pass ==="
    [ "$pass" = 0 ] || status=1
done
exit $status
