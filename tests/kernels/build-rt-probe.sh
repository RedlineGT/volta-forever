#!/bin/bash
# Build rt_probe (a small program of the CUDA runtime with a kernel of its own and calls to cuBLAS)
# with the two toolkits, for tests/gpu/run.sh. Needs g++. Output: $1 (directory).
set -euo pipefail
out=${1:?output directory}
src=$(cd "$(dirname "$0")" && pwd)/rt_probe.cu
W=$(cd "$(dirname "$0")/../.." && pwd)/toolkits
. "$(dirname "$0")/../../scripts/nvcc-host.sh"
mkdir -p "$out"
build() {  # build NAME TOOLKIT LIBDIR NVCC-FLAGS...
    local name=$1 tk=$2 lib=$3; shift 3
    # no rpath: run with LD_LIBRARY_PATH=<toolkit>/<libdir> so the libraries can be swapped
    nvcc_host_args "$tk/bin/nvcc"
    "$tk/bin/nvcc" "${NVCC_HOST_ARGS[@]}" -O2 "$@" -o "$out/rt_probe.$name" "$src" -L"$tk/$lib" -lcublas 2>&1 | grep -v 'deprecated-gpu-targets' || true
    [ -x "$out/rt_probe.$name" ] && echo "built rt_probe.$name"
}
build 129.sm70 $W/cuda-12.9.2 lib -arch=sm_70
build 135.sm75 $W/cuda-13.5.0 lib -arch=sm_75
build 135.sm75ptx $W/cuda-13.5.0 lib -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75
