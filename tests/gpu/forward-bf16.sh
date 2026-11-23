#!/bin/bash
# GPU tests of the bf16 cuBLAS variants (dist/forward/bf16tc, from scripts/build-forwarders.sh), on a
# machine with a V100 (review of the design, section 5: G1-G10):
#   tests/gpu/forward-bf16.sh [GPU] [CASE...]
# Each program of tests/gpu/forward/bf16-*.c is built against the CUDA 13.5 headers and libraries,
# then run with the proxy, the variants, the default forwarders and the CUDA 12.9.2 libraries in the
# library path. The variants load a test build of the runtime (-DVF_BF16_TEST: the same source, with
# counters that say which path served a call, injected failures and controls) placed beside copies of
# them; the release runtime runs the cases that need no counter. A case prints PASS or FAIL; exit
# status 1 on any FAIL. Builds into $VF_OUT (default $VF_TEST_ROOT).
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
T12=$W/toolkits/cuda-12.9.2
T13=$W/toolkits/cuda-13.5.0
out=${VF_OUT:-$VF_TEST_ROOT}
proxy=${VF_PROXY:-$W/dist/lib}
fwd=$W/dist/forward
src=$W/tests/gpu/forward
rt=$W/crates/vf-fwdgen/bf16tc
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${1:-0}
shift || true
. "$W/tests/gpu/cache.sh"
[ -e "$fwd/bf16tc/libvf_cublas_bf16.so" ] || { echo "missing: $fwd/bf16tc (run scripts/build-forwarders.sh)"; exit 2; }
mkdir -p "$out/bin" "$out/logs" "$out/bf16-test"

# The test runtime, beside copies of the variants (a variant loads the runtime of its own directory).
test_dir=$out/bf16-test
cp "$fwd/bf16tc/libcublas.so.13" "$fwd/bf16tc/libcublasLt.so.13" "$test_dir/"
gcc -shared -fPIC -O2 -std=gnu11 -Wall -Wextra -Wno-unused-parameter -Werror -fvisibility=hidden -DVF_BF16_TEST \
    -Wa,-I"$rt" -o "$test_dir/libvf_cublas_bf16.so" "$rt/runtime.c" "$rt/kernels.S" -Wl,-soname,libvf_cublas_bf16.so -Wl,-z,defs -ldl -lpthread ||
    { echo "FAIL  build of the test runtime"; exit 1; }

failed=0
build() {  # build NAME : compile tests/gpu/forward/NAME.c against the CUDA 13.5 headers and libraries
    gcc -O2 -Wall -Wno-deprecated-declarations -I"$T13/include" -o "$out/bin/$1" "$src/$1.c" \
        -L"$T13/lib" -Wl,-rpath-link,"$T13/lib" -L"$proxy" -lcublas -lcublasLt -lcudart -l:libcuda.so.1 -ldl -lm -lpthread ||
        { echo "FAIL  build of $1"; failed=1; return 1; }
}
run() {  # run NAME DIR [ENV...] : run with the variants of DIR first in the library path; output in $log
    local name=$1 dir=$2; shift 2
    log=$out/logs/forward-$name.log
    env LD_LIBRARY_PATH="$proxy:$dir:$fwd:$T12/lib:$T13/lib" VF_CUBLAS_BF16_TC=1 VF_LOG_LEVEL=2 "$@" \
        "$out/bin/$name" >"$log" 2>&1
    status=$?
}
verdict() {  # verdict NAME : PASS when the program printed "ok NAME" and exited with 0
    if [ "$status" = 0 ] && grep -qx "ok $1" "$log"; then echo "PASS  $1"; else echo "FAIL  $1 (exit status $status; $log)"; failed=1; fi
}
want() { [ ${#cases[@]} -eq 0 ] && return 0; local c; for c in "${cases[@]}"; do [ "$c" = "$1" ] && return 0; done; return 1; }
cases=("$@")

if want bf16-paths; then
    build bf16-paths && { run bf16-paths "$test_dir"; verdict bf16-paths; }
fi
exit $failed
