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
build() {  # build NAME [FLAGS...] : compile tests/gpu/forward/NAME.c against the CUDA 13.5 headers and libraries
    local name=$1; shift
    gcc -O2 "$@" -Wall -Wno-deprecated-declarations -I"$T13/include" -o "$out/bin/$name" "$src/$name.c" \
        -L"$T13/lib" -Wl,-rpath-link,"$T13/lib" -L"$proxy" -lcublas -lcublasLt -lcudart -l:libcuda.so.1 -ldl -lm -lpthread ||
        { echo "FAIL  build of $name"; failed=1; return 1; }
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
    for c in MULADD EPI_SEPARATE NO_ZERO_TERM ZERO_BETA0 ZERO_ALWAYS EPI_ALL SPLITK NAN_BITS; do
        (cd "$rt" && "$T12/bin/nvcc" -gencode arch=compute_70,code=sm_70 -Wno-deprecated-gpu-targets -O3 -fmad=false \
            -std=c++17 -DVF_KERNEL_CONTROL_$c -fatbin -o "$test_dir/control-$c.fatbin" kernels.cu) && controls+=("$c")
    done
fi

for t in bf16-paths bf16-arith bf16-graphs bf16-threads bf16-life bf16-reserve bf16-lt bf16-fallback bf16-s1; do
    if want $t; then
        build $t && { run $t "$test_dir"; verdict $t; }
    fi
done
# The tensor-core path's acceptance test (policy B, docs/emulations.md): every call it serves with k >= 2048
# against 12.9 and the exact product (special values, determinism, the error bound A, the bias bound B), and
# no call below the gate served. Controls: the gate moved to 1536, 1024 and 256 (VF_BF16_TC_MIN_K, read by test
# builds only, with the test's gate the same) must fail at a call of that k. VF_SAFE=1: no call served by the
# tensor-core path, every result 12.9's. -fopenmp shortens its host side only.
if want bf16-accept; then
    if build bf16-accept -fopenmp; then
        run bf16-accept "$test_dir" VF_CUBLAS_BF16_SKINNY=0 VF_ACCEPT_GATE=2048 VF_ACCEPT_EXPECT_GATE=1; verdict bf16-accept
        cp "$log" "$out/logs/forward-bf16-accept-main.log"
        for g in 1536 1024 256; do
            run bf16-accept "$test_dir" VF_CUBLAS_BF16_SKINNY=0 VF_BF16_TC_MIN_K=$g VF_ACCEPT_GATE=$g >/dev/null
            cp "$log" "$out/logs/forward-bf16-accept-gate$g.log"
            if [ "$status" != 0 ] && grep -q "^FAIL  [a-z0-9]* [NT][NT] [0-9]*x[0-9]*x$g " "$log"; then
                echo "PASS  bf16-accept control: the gate at $g fails at k = $g"
            else echo "FAIL  bf16-accept control: the gate at $g did not fail at k = $g ($out/logs/forward-bf16-accept-gate$g.log)"; failed=1; fi
        done
        run bf16-accept "$test_dir" VF_CUBLAS_BF16_SKINNY=1 VF_SAFE=1 VF_ACCEPT_MODE=safe
        cp "$log" "$out/logs/forward-bf16-accept-safe.log"
        if [ "$status" = 0 ] && grep -qx "ok bf16-accept safe" "$log"; then echo "PASS  bf16-accept safe (VF_SAFE=1)"
        else echo "FAIL  bf16-accept safe: under VF_SAFE=1 a call was served by the tensor-core path or lost 12.9's bits ($log)"; failed=1; fi
    fi
fi
# Lazy setup: one process per case, the first served call of the process's context in each.
if want bf16-lazy; then
    if build bf16-lazy; then
        for m in global threadlocal other-thread tc; do
            run bf16-lazy "$test_dir" VF_BF16_LAZY=$m
            if [ "$status" = 0 ] && grep -qx "ok bf16-lazy" "$log"; then echo "PASS  bf16-lazy $m"
            else echo "FAIL  bf16-lazy $m (exit status $status; $log)"; failed=1; fi
        done
    fi
fi
# VF_SAFE=1 with the tensor-core switch: the tensor-core path serves nothing, the skinny path serves.
if want bf16-safe; then
    build bf16-safe && { run bf16-safe "$test_dir" VF_SAFE=1; verdict bf16-safe; }
fi
# A language model's output layer (m = 151936): the skinny path alone forwards it; the tensor-core path
# serves it outside a capture, chunked, and forwards it captured.
if want bf16-lmhead; then
    if build bf16-lmhead; then
        for sw in skinny tc; do
            if [ $sw = skinny ]; then run bf16-lmhead "$test_dir" VF_CUBLAS_BF16_TC=0 VF_CUBLAS_BF16_SKINNY=1
            else run bf16-lmhead "$test_dir"; fi
            if [ "$status" = 0 ] && grep -qx "ok bf16-lmhead" "$log"; then echo "PASS  bf16-lmhead $sw"
            else echo "FAIL  bf16-lmhead $sw (exit status $status; $log)"; failed=1; fi
        done
    fi
fi
# G3 with the skinny path: its captured calls are kernel nodes of the runtime's module.
if want bf16-graphs-skinny; then
    build bf16-graphs && { run bf16-graphs "$test_dir" VF_BF16_GRAPHS_N=8; verdict bf16-graphs; }
fi
# G8: what the runtime is told, and how its kernels load, under the layer's settings: it must be told
# the real 7.0 (its file holds sm_70 machine code) and load its module, or every call would forward.
# Both module loading modes are required to pass: under CUDA_MODULE_LOADING=EAGER the layer gives no
# module for a library it cannot serve whole, so cuBLAS 12.9's cublasCreate gets past its sm_80 images.
if want bf16-capability; then
    if build bf16-arith; then
        for e in "VF_CAPABILITY=8.0" "VF_CAPABILITY=8.0 VF_CAPABILITY_ALL=1" "VF_CAPABILITY=12.0" \
                 "VF_CAPABILITY=12.0 VF_CAPABILITY_ALL=1" "VF_LAZY=0" "VF_ADAPT=0" "VF_TRANSLATE=0" \
                 "CUDA_MODULE_LOADING=EAGER" "CUDA_MODULE_LOADING=LAZY"; do
            # shellcheck disable=SC2086
            run bf16-arith "$test_dir" $e >/dev/null
            if [ "$status" = 0 ] && grep -qx "ok bf16-arith" "$log"; then echo "PASS  bf16-capability $e"
            else echo "FAIL  bf16-capability $e ($log)"; failed=1; fi
        done
    fi
fi
# G10: the checked-in fat binary and its manifest are what the source and the CUDA 12.9.2 compiler give
# (when the compiler is there; without it, vf-fwdgen's unit tests check the manifest's hashes).
if want bf16-kernels && [ -x "$T12/bin/nvcc" ]; then
    if NVCC="$T12/bin/nvcc" "$W/scripts/build-bf16-kernels.sh" --check; then echo "PASS  bf16-kernels"
    else echo "FAIL  bf16-kernels"; failed=1; fi
fi
# The skinny path alone: what it forwards to 12.9 rather than serve (the tensor-core path off).
if want bf16-paths-skinny; then
    build bf16-paths && { run bf16-paths "$test_dir" VF_CUBLAS_BF16_TC=0 VF_CUBLAS_BF16_SKINNY=1; verdict bf16-paths; }
fi
exit $failed
