#!/bin/bash
# Triton's kernels through the layer against Triton's interpreter, and the layer's two routes against
# each other: the quick subset of the corpus of tests/gpu/triton_corpus.py (about 170 cases: element-
# wise operations in every type, conversions, reductions, scans, sorts, tl.dot in several types,
# precisions, tiles, warps and stages, flash attention, masked and gathered memory, block pointers,
# atomics, loops, grids, random numbers), run by tests/gpu/triton-diff.py (read its head):
#   - the inputs of each case, made once from its seed and saved with the reference (run/ref/inputs),
#     which the interpreter and both routes take: the report gives their digest with a difference;
#   - the reference: each case under TRITON_INTERPRET=1 on the CPU, in processes without a GPU (or a
#     computation with PyTorch on the CPU where the case names one);
#   - route `sass`: as a program runs it: Triton is told 7.5 and makes sm_75 machine code, which the
#     layer translates; 5 runs of each case on fresh copies of its inputs, which must give the same
#     bits;
#   - route `ptx`: the same compiled kernels loaded from the PTX that Triton made them from (the
#     layer retargets it, the driver's compiler makes sm_70 code of it), 2 runs;
#   - every case must agree with its reference (bit for bit, or within the tolerance the case says)
#     and the two routes must give the same bits. With VF_LOG_LEVEL=2 the lines of the layer say
#     which route each module took; a module that the sass route did not translate fails.
# The whole corpus (3,300 cases) is run by hand: `triton-diff.py interp/gpu/report --level full`.
#
# Run it on a machine with a V100, the GPU chosen by CUDA_VISIBLE_DEVICES. It needs $VF_PROXY, the
# PyTorch site (PyTorch 2.14.0+cu130 with its Triton 3.8), numpy (VF_TRITON_EXTRA, below) and Python
# 3.12 headers for Triton's C stub; or VF_TRITON_PYTHON, a Python that has all three, used as it is.
# Settings: VF_TRITON_LEVEL (quick, or full), VF_TRITON_FILTER (a regular expression of case names),
# VF_TRITON_JOBS (processes of the interpreter, default 8), VF_TRITON_VENDOR=1 (also the kernels of
# vLLM or SGLang, whichever VF_TRITON_PYTHON has: tests/gpu/triton_vendor.py). Exit status 1 when a
# case fails.
# With VF_TRITON_TC set (INSTALL.md), the layer's python/ goes first in PYTHONPATH, as the launcher
# puts it: the kernels whose products are of the types it names are compiled for sm_80, and the
# layer runs the PTX that their cubins keep (route sass); the verdicts say so.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
vf_clear_settings VF_TRITON_TC VF_TRITON_TC_TIMEOUT VF_TRITON_TC_START VF_TRITON_TC_JOBS VF_TRITON_TC_WAIT \
    VF_TRITON_TC_IDLE
out=${VF_OUT:-$VF_TEST_ROOT}

proxy=${VF_PROXY:-$W/dist/lib}
torch_site=${VF_TORCH:-$VF_TEST_ROOT/torch-cu130/site}
pydev=${VF_PYTHON_DEV:-$VF_TEST_ROOT/python-dev}
map=${VF_AUDIT_MAP_FILE:-$out/torch-audit.map}
python=${VF_TRITON_PYTHON:-python3}
work=$out/triton-test
rm -rf "$work/run"
mkdir -p "$work/run" "$work/tmp" "$work/.cache" "$work/triton"

export HOME=$work TMPDIR=$work/tmp XDG_CACHE_HOME=$work/.cache TRITON_CACHE_DIR=$work/triton
export CUDA_DEVICE_ORDER=PCI_BUS_ID
. "$W/tests/gpu/cache.sh"
export PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 DO_NOT_TRACK=1
# (The interpreter needs numpy, which the PyTorch wheel does not bring: VF_TRITON_EXTRA, a directory
# with numpy installed, `pip install --target DIR numpy`.)
extra=${VF_TRITON_EXTRA:-$VF_TEST_ROOT/triton-extra/site}
if [ -z "${VF_TRITON_PYTHON:-}" ]; then
    export PYTHONPATH="$torch_site"
    [ -d "$extra" ] && export PYTHONPATH="$torch_site:$extra"
fi
case ${VF_TRITON_TC:-0} in
    ''|0) ;;
    *) export PYTHONPATH="$W/python${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
export LD_LIBRARY_PATH="$proxy" LIBRARY_PATH="$proxy"
[ -f "$map" ] && export LD_AUDIT="$proxy/libvf_audit.so" VF_AUDIT_MAP_FILE="$map"
[ -d "$pydev/usr/include/python3.12" ] && export CPATH="$pydev/usr/include/python3.12:$pydev/usr/include"
# Level 2: the lines that say which route each module took, in a file the cases are matched with.
export VF_LOG_LEVEL=2 VF_LOG=$work/run/vf.log

args=(--work "$work/run" --level "${VF_TRITON_LEVEL:-quick}" --runs 5 --jobs "${VF_TRITON_JOBS:-8}")
[ -n "${VF_TRITON_FILTER:-}" ] && args+=(--filter "$VF_TRITON_FILTER")
[ "${VF_TRITON_VENDOR:-0}" = 1 ] && args+=(--vendor)
"$python" -s "$W/tests/gpu/triton-diff.py" all "${args[@]}"
status=$?
echo "=== triton: status $status (verdicts in $work/run/verdicts.json, the layer's lines in $work/run/vf.log) ==="
exit $status
