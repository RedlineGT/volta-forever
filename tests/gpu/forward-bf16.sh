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
    if [ -n "${VF_BF16_SANITIZE:-}" ]; then
        # NVIDIA's compute-sanitizer (12.9.2), without the proxy: the runtime's kernels and cuBLAS
        # 12.9's are sm_70 machine code, which the sanitizer checks as they are.
        env LD_LIBRARY_PATH="$dir:$fwd:$T12/lib:$T13/lib" VF_CUBLAS_BF16_TC=1 VF_LOG_LEVEL=2 "$@" \
            "$T12/compute-sanitizer/compute-sanitizer" --tool "$VF_BF16_SANITIZE" --leak-check no --error-exitcode 99 \
            "$out/bin/$name" >"$log" 2>&1
    else
        env LD_LIBRARY_PATH="$proxy:$dir:$fwd:$T12/lib:$T13/lib" VF_CUBLAS_BF16_TC=1 VF_LOG_LEVEL=2 "$@" \
            "$out/bin/$name" >"$log" 2>&1
    fi
    status=$?
    cat "$log"
}
verdict() {  # verdict NAME : PASS when the program printed "ok NAME" and exited with 0
    if [ "$status" = 0 ] && grep -qx "ok $1" "$log"; then echo "PASS  $1"; else echo "FAIL  $1 (exit status $status; $log)"; failed=1; fi
}
want() { [ ${#cases[@]} -eq 0 ] && return 0; local c; for c in "${cases[@]}"; do [ "$c" = "$1" ] && return 0; done; return 1; }
cases=("$@")

# Fat binaries with a kernel control (VF_BF16_FATBIN=<file> makes the test runtime load one), when
# the CUDA 12.9.2 compiler is there: the arithmetic tests must fail with them.
controls=()
if [ -x "$T12/bin/nvcc" ]; then
    for c in MULADD EPI_SEPARATE NO_ZERO_TERM EPI_ALL SPLITK; do
        (cd "$rt" && "$T12/bin/nvcc" -gencode arch=compute_70,code=sm_70 -Wno-deprecated-gpu-targets -O3 -fmad=false \
            -std=c++17 -DVF_KERNEL_CONTROL_$c -fatbin -o "$test_dir/control-$c.fatbin" kernels.cu) && controls+=("$c")
    done
fi

for t in bf16-paths bf16-arith; do
    if want $t; then
        build $t && { run $t "$test_dir"; verdict $t; }
    fi
done
# The skinny path alone: what it forwards to 12.9 rather than serve (the tensor-core path off).
if want bf16-paths-skinny; then
    build bf16-paths && { run bf16-paths "$test_dir" VF_CUBLAS_BF16_TC=0 VF_CUBLAS_BF16_SKINNY=1; verdict bf16-paths; }
fi
exit $failed
