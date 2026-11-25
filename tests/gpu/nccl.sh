#!/bin/bash
# NCCL's collectives through PyTorch on two V100s, through the layer:
#   tests/gpu/nccl.sh [cuda13|cuda12] [--bench]
# On the first two GPUs that CUDA_VISIBLE_DEVICES leaves visible (PCI order). Runs
# tests/gpu/nccl-collectives.py (which says what each case checks) with the stock wheel of PyTorch
# in $VF_TORCH (as tests/gpu/torch.sh), one process for each GPU, every result compared with the
# CPU's, exactly.
#   cuda13 (the default): the NCCL of the wheel (nvidia-nccl-cu13), PTX for later GPUs only, which
#          the layer adapts: its kernels are given with reported traps in place of the multicast
#          instructions they hold (LIMITS.md, "Several GPUs and NCCL"). A refused NCCL kernel, or
#          a line of the driver for an assertion of the layer, fails the suite.
#   cuda12: the CUDA 12 build of NCCL, which has code for Volta, put in place of the wheel's by
#          the audit map (as `vf fetch --only libnccl` and the launcher do): $VF_NCCL12 or
#          toolkits/cuda-12.9.2/lib/libnccl.so.2.
# --bench also times an all-reduce of 64 MiB and one of 8 bytes. Fewer than two GPUs visible: status
# 2. The first run with cuda13 compiles NCCL's adapted code: about 15 minutes of one CPU core for
# each process (LIMITS.md, "Several GPUs and NCCL"); CUDA_CACHE_PATH keeps it, with the larger
# CUDA_CACHE_MAXSIZE set here (the driver's default, 256 MiB, does not hold it).
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
# NCCL's adapted code is large: the driver keeps its compilation only in a cache that holds it.
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
vf_layer "$proxy" "$map"
NCCL_DEBUG=${NCCL_DEBUG:-WARN} VF_LOG="$layer" python3 "$W/tests/gpu/nccl-collectives.py" "$@" >"$log" 2>&1
status=$?
cat "$log"
[ $status = 2 ] && exit 2
failed=$status
refused=$(grep "cannot run on this GPU and is not given to the program" "$layer" | grep -c "kernel _Z[0-9]*ncclDevKernel_\|kernel ncclDevKernel_")
if [ "$refused" != 0 ]; then
    echo "FAIL  $refused kernels of NCCL's collectives were refused ($layer)"
    failed=1
fi
if grep -q "^volta-forever:" "$log"; then
    echo "FAIL  a reported trap of the layer was reached ($log)"
    failed=1
fi
given=$(grep -c "is given with .* replaced by a reported trap" "$layer")
echo "the layer's lines: $layer ($given kernels said to be given with reported traps, $(grep -c "cannot run on this GPU and is not given" "$layer") refused)"
exit $failed
