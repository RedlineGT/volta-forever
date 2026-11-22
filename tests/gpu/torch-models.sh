#!/bin/bash
# Whole models of PyTorch through the layer, on a machine with a V100:
#   tests/gpu/torch-models.sh [GPU [-v] [PATTERN...]]
# Runs tests/gpu/torch-models.py (which says what a case is and how it ends) with the stock wheel of
# tests/gpu/torch.sh ($VF_TORCH, default $VF_TEST_ROOT/torch-cu130/site), the forwarding
# libraries in front of its cuBLAS when the map is at $VF_OUT/torch-audit.map (or
# $VF_AUDIT_MAP_FILE). With VF_NATIVE=1: the same cases with the wheel that has sm_70 code
# ($VF_TORCH_NATIVE, default $VF_TEST_ROOT/torch-cu126/site, see tests/gpu/torch-bench.sh) and
# without the layer: what a V100 does by itself, to compare with. GPU is the PCI-order index
# (default 0). (The wheels are looked for under VF_TEST_ROOT, as by every script of tests/gpu;
# VF_OUT is where this run writes.)
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
out=${VF_OUT:-$VF_TEST_ROOT}
gpu=${1:-0}
[ $# -gt 0 ] && shift
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu
. "$W/tests/gpu/cache.sh"
export VF_OUT=$out
if [ "${VF_NATIVE:-0}" = 1 ]; then
    site=${VF_TORCH_NATIVE:-$VF_TEST_ROOT/torch-cu126/site}
    [ -d "$site/torch" ] || { echo "missing: PyTorch in $site (the head of tests/gpu/torch-bench.sh says how to put it there)"; exit 2; }
    PYTHONPATH=$site python3 "$W/tests/gpu/torch-models.py" "$@"
    exit
fi
site=${VF_TORCH:-$VF_TEST_ROOT/torch-cu130/site}
[ -d "$site/torch" ] || { echo "missing: PyTorch in $site (the head of tests/gpu/torch.sh says how to put it there)"; exit 2; }
proxy=${VF_PROXY:-$W/dist/lib}
map=$(vf_torch_map "$out")
vf_layer "$proxy" "$map"
PYTHONPATH=$site python3 "$W/tests/gpu/torch-models.py" "$@"
