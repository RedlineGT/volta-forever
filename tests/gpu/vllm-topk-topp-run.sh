#!/bin/bash
# Run the _topk_topp_kernel correctness test against a CPU reference.
# Two passes: first dumps cubins and translates them with VF_BORROW=2,
# then runs the test with the hook loading the level-3 translations.
#
# This is run INSIDE a gpu-run.sh call (it expects $VF_GPU to be set).
# It needs: $VF_PROXY, Python 3.12 headers (for Triton's C stub),
# and the vLLM/torch sites.
# The exit status is the test's: of pass 2 when there is one, else of pass 1.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
host=$(hostname)
out=${VF_OUT:-$VF_TEST_ROOT}

proxy=${VF_PROXY:-$W/dist/lib}
vllm_site=${VF_VLLM_SITE:-$VF_TEST_ROOT/vllm/site}
torch_site=${VF_TORCH:-$VF_TEST_ROOT/torch-cu130/site}
extra=${VF_VLLM_EXTRA:-$VF_TEST_ROOT/vllm-extra/site}
pydev=${VF_PYTHON_DEV:-$VF_TEST_ROOT/python-dev}
map=${VF_AUDIT_MAP_FILE:-$out/torch-audit.map}

cache=$out/vllm-topk-cache
dump=${VF_VLLM_TOPK_DUMP:-$VF_TEST_ROOT/vllm-topk/dump}
hook=${VF_VLLM_TOPK_HOOK:-$VF_TEST_ROOT/vllm-topk/hook-dir}
mkdir -p "$cache/tmp" "$cache/.cache" "$cache/triton" "$cache/hf" "$dump"

export HOME=$cache TMPDIR=$cache/tmp XDG_CACHE_HOME=$cache/.cache
export TRITON_CACHE_DIR=$cache/triton HF_HOME=$cache/hf
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# CUDA_VISIBLE_DEVICES is already set by gpu-run.sh
source "$W/tests/gpu/cache.sh"
export PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1

pp="$vllm_site:$torch_site"
[ -d "$extra" ] && pp="$extra:$pp"
export PYTHONPATH="$pp"
export LD_LIBRARY_PATH="$proxy"
[ -f "$map" ] && export LD_AUDIT="$proxy/libvf_audit.so" VF_AUDIT_MAP_FILE="$map"
[ -d "$pydev/usr/include/python3.12" ] && export CPATH="$pydev/usr/include/python3.12:$pydev/usr/include"
export LIBRARY_PATH="$proxy"
export VF_BORROW=1
export VF_LOG_LEVEL=1

echo "=== Pass 1: compile kernels, dump cubins, find refusals ==="
rm -rf "$dump"/*
export VF_TRANSLATE_DUMP="$dump"
export VF_DEBUG_TRANSLATED="$hook"
# Run the test; it will fail on refused kernels but that's expected (pass 2 then has the verdict)
python3 -s "$W/tests/gpu/vllm-topk-topp-test.py" 2>&1
first=$?
unset VF_TRANSLATE_DUMP

echo ""
echo "=== Pass 1.5: translate every dumped cubin that was refused ==="
translated=0
for f in "$dump"/*.sm_75.cubin; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    # Try translating with VF_BORROW=1 first (what the proxy does)
    result=$(VF_BORROW=1 VF_BORROW_SAY=1 python3 -I "$W/reference/translate.py" "$f" /dev/null "$cache/tmp-map" 2>&1)
    if echo "$result" | grep -q "[1-9][0-9]* refused"; then
        # This cubin has refused functions; translate with VF_BORROW=5
        echo "  refused at borrow=1: $base"
        echo "  translating with VF_BORROW=5..."
        result2=$(VF_BORROW=5 VF_BORROW_SAY=1 python3 -I "$W/reference/translate.py" "$f" "$cache/out.cubin" "$cache/out.map" 2>&1)
        echo "  $result2"
        if echo "$result2" | grep -q "[1-9][0-9]* refused"; then
            echo "  STILL REFUSED at borrow=5 -- cannot translate"
            continue
        fi
        # Compute FNV-1a hash
        hash=$(python3 -I -c "
data = open('$f','rb').read()
h = 0xcbf29ce484222325
for b in data:
    h = ((h ^ b) * 0x100000001b3) & 0xffffffffffffffff
print(f'{h:016x}')
")
        cp "$cache/out.cubin" "$hook/$hash.cubin"
        touch "$hook/$hash.refused"  # empty: no functions refused
        echo "  placed: $hook/$hash.cubin"
        translated=$((translated + 1))
    fi
done
rm -f "$cache/out.cubin" "$cache/out.map" "$cache/tmp-map" "$cache/tmp-map.lives"
echo "  $translated cubins translated and placed in hook directory"

if [ $translated -gt 0 ]; then
    echo ""
    echo "=== Pass 2: clear proxy cache and run test with all translations available ==="
    rm -rf "$VF_PROXY_CACHE"/*
    python3 -s "$W/tests/gpu/vllm-topk-topp-test.py" 2>&1
    exit $?
fi
echo ""
echo "=== No new translations needed; pass 1 results are final ==="
exit $first
