#!/bin/bash
# NCCL's collectives through PyTorch on two V100s, through the layer:
#   tests/gpu/nccl.sh [cuda13|cuda12] [--bench]
# On the first two GPUs that CUDA_VISIBLE_DEVICES leaves visible (PCI order). Runs
# tests/gpu/nccl-collectives.py (which says what each case checks) with the stock wheel of PyTorch
# in $VF_TORCH (as tests/gpu/torch.sh), one process for each GPU, every value, integer or float,
# compared with the CPU's exactly: with NCCL's own choice of algorithm and protocol, then with each
# pinned (Ring and Tree, Simple, LL and LL128: --protocols), so that the functions of each run.
#   cuda13 (the default): the NCCL of the wheel (nvidia-nccl-cu13), PTX for later GPUs only, which
#          the layer adapts: its kernels are given with reported traps in place of the multicast
#          instructions they hold (LIMITS.md, "Several GPUs and NCCL"). A refused NCCL kernel, or
#          a line of the driver for an assertion of the layer, fails the suite.
#   cuda12: the CUDA 12 build of NCCL, which has code for Volta, put in place of the wheel's by
#          the audit map (as `vf fetch --only libnccl` and the launcher do): $VF_NCCL12 or
#          toolkits/cuda-12.9.2/lib/libnccl.so.2.
# --bench also times an all-reduce of 64 MiB and one of 8 bytes. Fewer than two GPUs visible: status
# 2. The first run with cuda13 compiles NCCL's adapted code: about 15 minutes of one CPU core for
# each process (LIMITS.md, "Several GPUs and NCCL"); CUDA_CACHE_PATH keeps it (about 383 MB of a
# cache that all the driver's compiled code shares; its default limit, 1 GiB, holds it).
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
vf_clear_settings
which=cuda13
case ${1:-} in cuda13 | cuda12) which=$1; shift ;; esac
site=${VF_TORCH:-$VF_TEST_ROOT/torch-cu130/site}
[ -d "$site/torch" ] || { echo "missing: PyTorch in $site (tests/gpu/torch.sh says how to put it there)"; exit 2; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=$site
. "$W/tests/gpu/cache.sh"
export VF_OUT=${VF_OUT:-$VF_TEST_ROOT}
out=$VF_OUT/nccl
mkdir -p "$out"
# (A cache shared with the suites' other compiled code: room for all of it.)
export CUDA_CACHE_MAXSIZE=${CUDA_CACHE_MAXSIZE:-4294967296}
proxy=${VF_PROXY:-$W/dist/lib}
map=$out/$which-audit.map
cat "$(vf_torch_map "$VF_OUT")" >"$map"
case $which in
    cuda13) ;;
    cuda12)
        nccl12=${VF_NCCL12:-$W/toolkits/cuda-12.9.2/lib/libnccl.so.2}
        [ -f "$nccl12" ] || { echo "missing: the CUDA 12 NCCL ($nccl12; VF_NCCL12)"; exit 2; }
        echo "libnccl.so.2=$nccl12" >>"$map"
        ;;
    *) echo "usage: $0 [cuda13|cuda12] [--bench]"; exit 2 ;;
esac
case ${1:-} in "" | --bench) ;; *) echo "usage: $0 [cuda13|cuda12] [--bench]"; exit 2 ;; esac
log=$out/$which.log layer=$out/$which.layer.log
: >"$layer"
rm -f "$out/$which".nccl.*.log
vf_layer "$proxy" "$map"
# NCCL's own lines, with what it ran for each collective (its algorithm and protocol), in a file of
# each process.
NCCL_DEBUG=${NCCL_DEBUG:-INFO} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,ENV,TUNING} \
    NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE:-$out/$which.nccl.%p.log} VF_LOG="$layer" \
    python3 "$W/tests/gpu/nccl-collectives.py" --protocols "$@" >"$log" 2>&1
status=$?
cat "$log"
echo "what NCCL ran (collective, algorithm, protocol: launches over the ranks):"
cat "$out/$which".nccl.*.log 2>/dev/null | grep -oE "INFO [A-Za-z]+: [0-9]+ Bytes -> Algo [A-Z_]+ proto [A-Z0-9]+" |
    awk '{print $2, $7, $9}' | sort | uniq -c | sed 's/^/  /'
[ $status = 2 ] && exit 2
failed=$status
# Each pinned pair that NCCL took ran its all-reduces with that algorithm and protocol.
ran=$(cat "$out/$which".nccl.*.log 2>/dev/null | grep -oE "INFO AllReduce: [0-9]+ Bytes -> Algo [A-Z_]+ proto [A-Z0-9]+")
for algo in Ring Tree; do
    for proto in Simple LL LL128; do
        grep -q "^NOT TAKEN  NCCL_ALGO=$algo NCCL_PROTO=$proto:" "$log" && continue
        if ! grep -q "Algo ${algo^^} proto ${proto^^}$" <<<"$ran"; then
            echo "FAIL  no all-reduce ran with $algo and $proto, which the pass of that pair asked for"
            failed=1
        fi
    done
done
refused=$(grep "cannot run on this GPU and is not given to the program" "$layer" | grep -c "kernel _Z[0-9]*ncclDevKernel_\|kernel ncclDevKernel_")
if [ "$refused" != 0 ]; then
    echo "FAIL  $refused kernels of NCCL's collectives were refused ($layer)"
    failed=1
fi
if grep -q "^volta-forever:" "$log"; then
    echo "FAIL  a reported trap of the layer was reached ($log)"
    failed=1
fi
images=$(grep -Ec "(kernel is|kernels are) given with instructions that cannot run on this GPU replaced by reported traps" "$layer")
large=$(grep -c "MB of PTX that the layer made of it" "$layer")
echo "the layer's lines: $layer ($images lines of images given with reported traps, $large of a large compilation, $(grep -c "cannot run on this GPU and is not given" "$layer") refused kernels)"
exit $failed
