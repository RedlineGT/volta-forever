#!/bin/bash
# Operations of PyTorch through the layer, on a machine with a V100:
#   tests/gpu/torch.sh [GPU [-v] [PATTERN...]]
# Runs tests/gpu/torch-ops.py (which says what a case is and how it ends) with a stock wheel of
# PyTorch: machine code for sm_75 and later, no PTX, so that every kernel it launches on a V100 was
# translated by the layer. The wheel and what it needs (5.4 GB) are to be in $VF_TORCH (default
# $VF_TEST_ROOT/torch-cu130/site), put there, from a machine with the same Python, by
#   python3 -m pip install --target DIR --index-url https://download.pytorch.org/whl/cu130 \
#       --extra-index-url https://pypi.org/simple torch==2.14.0
# GPU is the PCI-order index (default 0).
#
# PyTorch finds NVIDIA's libraries by a path recorded in its own, not by LD_LIBRARY_PATH: to put the
# forwarding libraries (dist/forward) in front of the wheel's, they are named in a map
# that the loader's audit library reads. For cuBLAS, whose own matrix products the translator does
# not take (the paths are those of this tree):
#   libcublas.so.13=.../dist/forward/libcublas.so.13
#   libcublasLt.so.13=.../dist/forward/libcublasLt.so.13
#   libcublas.so.12=.../toolkits/cuda-12.9.2/lib/libcublas.so.12
#   libcublasLt.so.12=.../toolkits/cuda-12.9.2/lib/libcublasLt.so.12
# The map is $VF_AUDIT_MAP_FILE, or $VF_OUT/torch-audit.map, which is written with those four
# lines when it does not exist (tests/gpu/env.sh).
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
site=${VF_TORCH:-$VF_TEST_ROOT/torch-cu130/site}
gpu=${1:-0}
[ $# -gt 0 ] && shift
[ -d "$site/torch" ] || { echo "missing: PyTorch in $site (the head of $0 says how to put it there)"; exit 2; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$site
. "$W/tests/gpu/cache.sh"
# (The proxy's log of a run goes with the other logs, not to the node's /tmp.)
export VF_OUT=${VF_OUT:-$VF_TEST_ROOT}
proxy=${VF_PROXY:-$W/dist/lib}
map=$(vf_torch_map "$VF_OUT")
python3 "$W/tests/gpu/torch-ops.py" --comparison || exit 1
vf_layer "$proxy" "$map"
python3 "$W/tests/gpu/torch-ops.py" "$@"
