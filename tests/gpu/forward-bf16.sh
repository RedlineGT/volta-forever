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
vf_clear_settings
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
# the CUDA 12.9.2 compiler is there: the arithmetic tests must fail with them, bf16-fallback with NO_STATS
# (the exit report's counters left at zero) and bf16-verify with PERTURB (the skinny kernel's element (0, 0) changed).
controls=()
if [ -x "$T12/bin/nvcc" ]; then
    . "$W/scripts/nvcc-host.sh"
    nvcc_host_args "$T12/bin/nvcc"
    for c in MULADD EPI_SEPARATE NO_ZERO_TERM ZERO_BETA0 ZERO_ALWAYS EPI_ALL SPLITK NAN_BITS NO_STATS PERTURB; do
        (cd "$rt" && "$T12/bin/nvcc" "${NVCC_HOST_ARGS[@]}" -gencode arch=compute_70,code=sm_70 -Wno-deprecated-gpu-targets -O3 -fmad=false \
            -std=c++17 -DVF_KERNEL_CONTROL_$c -fatbin -o "$test_dir/control-$c.fatbin" kernels.cu) && controls+=("$c")
    done
fi

for t in bf16-paths bf16-arith bf16-graphs bf16-threads bf16-life bf16-reserve bf16-lt bf16-fallback bf16-s1; do
    if want $t; then
        build $t && { run $t "$test_dir"; verdict $t; }
    fi
done
# The tensor-core path's acceptance test (policy B, docs/emulations.md): every call it serves with k >= 768
# against 12.9 and the exact product (special values, determinism, the error bound A, the bias bound B), in fp32
# and bf16 output, and no call below the gate served, every call from the gate on served. Controls, each of which
# must fail: the gate moved to 512 and 256 (VF_BF16_TC_MIN_K, read by test builds only, with the test's gate the
# same), at a call of that k of another family than lognpos and advpos; VF_ACCEPT_STRICT=1 (the bias of its four
# documented exceptions judged from the gate too), at lognpos and advpos at k = 768; the tensor-core path off, as
# not serving. VF_SAFE=1: no call served by the tensor-core path, every result 12.9's.
# -fopenmp shortens its host side only.
if want bf16-accept; then
    if build bf16-accept -fopenmp; then
        run bf16-accept "$test_dir" VF_CUBLAS_BF16_SKINNY=0 VF_ACCEPT_GATE=768 VF_ACCEPT_EXPECT_GATE=1; verdict bf16-accept
        cp "$log" "$out/logs/forward-bf16-accept-main.log"
        for g in 512 256; do
            run bf16-accept "$test_dir" VF_CUBLAS_BF16_SKINNY=0 VF_BF16_TC_MIN_K=$g VF_ACCEPT_GATE=$g >/dev/null
            cp "$log" "$out/logs/forward-bf16-accept-gate$g.log"
            # at k = $g, rows of other families than the two exceptions judged from 1024 (lognpos, advpos)
            if [ "$status" != 0 ] && grep "^FAIL  [a-z0-9]* [NT][NT] [0-9]*x[0-9]*x$g " "$log" | grep -qv "^FAIL  \(lognpos\|advpos\) "; then
                echo "PASS  bf16-accept control: the gate at $g fails at k = $g (other families than lognpos and advpos)"
            else echo "FAIL  bf16-accept control: the gate at $g did not fail at k = $g ($out/logs/forward-bf16-accept-gate$g.log)"; failed=1; fi
        done
        # Control: the four documented exceptions removed (VF_ACCEPT_STRICT=1): a bias from the gate on must fail, and
        # lognpos and advpos at the gate.
        run bf16-accept "$test_dir" VF_CUBLAS_BF16_SKINNY=0 VF_ACCEPT_GATE=768 VF_ACCEPT_STRICT=1 >/dev/null
        cp "$log" "$out/logs/forward-bf16-accept-strict.log"
        if [ "$status" != 0 ] && grep -q "^FAIL  lognpos TN 1024x128x768 fp32" "$log" && grep -q "^FAIL  advpos TN 1024x128x768 fp32" "$log"; then
            echo "PASS  bf16-accept control: VF_ACCEPT_STRICT=1 fails (the exceptions judged from the gate: lognpos and advpos at k = 768)"
        else echo "FAIL  bf16-accept control: VF_ACCEPT_STRICT=1 did not fail ($out/logs/forward-bf16-accept-strict.log)"; failed=1; fi
        # Control: the tensor-core path off (the skinny path on, so that the variants load the test runtime and its
        # counters), the gate expected: every call from the gate on must then fail as not served.
        run bf16-accept "$test_dir" VF_CUBLAS_BF16_TC=0 VF_CUBLAS_BF16_SKINNY=1 VF_ACCEPT_GATE=768 VF_ACCEPT_EXPECT_GATE=1 >/dev/null
        cp "$log" "$out/logs/forward-bf16-accept-tc-off.log"
        if [ "$status" != 0 ] && grep -q "^FAIL  .*: not served by the tensor-core path at k >= the gate" "$log"; then
            echo "PASS  bf16-accept control: with the tensor-core path off, the calls from the gate on are not served and it fails"
        else echo "FAIL  bf16-accept control: the tensor-core path off did not fail as not served ($out/logs/forward-bf16-accept-tc-off.log)"; failed=1; fi
        run bf16-accept "$test_dir" VF_CUBLAS_BF16_SKINNY=1 VF_SAFE=1 VF_ACCEPT_MODE=safe
        cp "$log" "$out/logs/forward-bf16-accept-safe.log"
        if [ "$status" = 0 ] && grep -qx "ok bf16-accept safe" "$log"; then echo "PASS  bf16-accept safe (VF_SAFE=1)"
        else echo "FAIL  bf16-accept safe: under VF_SAFE=1 a call was served by the tensor-core path or lost 12.9's bits ($log)"; failed=1; fi
    fi
fi
# Verify mode (VF_CUBLAS_BF16_VERIFY=1): every call the skinny path serves, and the tensor-core path's fallback
# elements, also computed by cuBLAS 12.9 and compared on the device, eager and captured; the report at exit says
# that nothing differed, once. Then the skinny path's captured calls in bf16-graphs' cases (two instantiations, a
# child graph, device launch, a conditional node's body, an update from a recapture) with their verification nodes:
# each step must do what it does with 12.9 alone, and nothing may differ.
if want bf16-verify; then
    if build bf16-verify; then
        run bf16-verify "$test_dir" VF_CUBLAS_BF16_VERIFY=1
        tot=$(grep -c "verify: the skinny path's calls against cuBLAS 12.9's, bit for bit: [1-9][0-9]* calls verified (eager calls and graph replays), in [0-9]* call signatures; 0 differed, in 0 elements" "$log")
        dif=$(grep -c "verify: DIFFERED" "$log")
        if [ "$status" = 0 ] && grep -qx "ok bf16-verify" "$log" && [ "$tot" = 1 ] && [ "$dif" = 0 ]; then
            echo "PASS  bf16-verify (the report at exit: nothing differed, said once)"
        else echo "FAIL  bf16-verify (exit status $status; report's totals $tot times, $dif signatures that differed; $log)"; failed=1; fi
    fi
    # Two threads' calls on one stream (a program's stream, the legacy stream, the per-thread default stream) and on a
    # stream each: every call verified, none reported as differing; with the control verify-unlocked (no lock across
    # a call's launches) the shared streams' calls are reported as differing, and the test fails.
    if build bf16-verify-threads; then
        run bf16-verify-threads "$test_dir" VF_CUBLAS_BF16_VERIFY=1
        tot=$(grep -c "verify: the skinny path's calls against cuBLAS 12.9's, bit for bit: [1-9][0-9]* calls verified (eager calls and graph replays), in [0-9]* call signatures; 0 differed, in 0 elements" "$log")
        dif=$(grep -c "verify: DIFFERED" "$log")
        if [ "$status" = 0 ] && grep -qx "ok bf16-verify-threads" "$log" && [ "$tot" = 1 ] && [ "$dif" = 0 ]; then
            echo "PASS  bf16-verify-threads (the report at exit: nothing differed, said once)"
        else echo "FAIL  bf16-verify-threads (exit status $status; report's totals $tot times, $dif signatures that differed; $log)"; failed=1; fi
        run bf16-verify-threads "$test_dir" VF_CUBLAS_BF16_VERIFY=1 VF_BF16_CONTROL=verify-unlocked >/dev/null
        cp "$log" "$out/logs/forward-bf16-verify-threads-unlocked.log"
        if [ "$status" != 0 ] && grep -q "^FAIL .*: verify mode reports [1-9][0-9]* calls differing" "$log" && grep -q "verify: DIFFERED" "$log"; then
            echo "PASS  bf16-verify-threads control: without the stream's lock (verify-unlocked) it fails there"
        else echo "FAIL  bf16-verify-threads control verify-unlocked: did not fail ($out/logs/forward-bf16-verify-threads-unlocked.log)"; failed=1; fi
    fi
    if build bf16-graphs; then
        run bf16-graphs "$test_dir" VF_BF16_GRAPHS_N=8 VF_CUBLAS_BF16_VERIFY=1
        cp "$log" "$out/logs/forward-bf16-graphs-verify.log"
        tot=$(grep -c "verify: the skinny path's calls against cuBLAS 12.9's, bit for bit: [1-9][0-9]* calls verified (eager calls and graph replays), in [0-9]* call signatures; 0 differed, in 0 elements" "$log")
        if [ "$status" = 0 ] && grep -qx "ok bf16-graphs" "$log" && [ "$tot" = 1 ]; then
            echo "PASS  bf16-graphs with the skinny path captured, verify mode (each graph case as with 12.9 alone; nothing differed)"
        else echo "FAIL  bf16-graphs with the skinny path captured, verify mode (exit status $status; report's totals $tot times; $out/logs/forward-bf16-graphs-verify.log)"; failed=1; fi
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
# The tensor-core path's layout rule, tall rule and wide rule: a fine-tuning step's input and weight gradients (NN,
# NT), an output layer up to the tall rule's n (TN, m = 151936) and a wide down projection past the wide rule's n go to
# 12.9 with its bits; the TN products of the step, the output layer one column more and the wide down projection at
# the wide rule's n to the tensor-core path. Each control that removes a rule (test builds) must make it fail there;
# a bound the runtime does not report (VF_RULES_NO_BOUND=5: the long-k branch's m read as -1, as a runtime without
# the branch reports it) must fail its check with exit status 1, not crash the test.
if want bf16-rules; then
    if build bf16-rules; then
        run bf16-rules "$test_dir"; verdict bf16-rules
        for c in tc-layouts:layout tc-tall:"tall rule" tc-wide:"wide rule"; do
            run bf16-rules "$test_dir" VF_BF16_CONTROL=${c%%:*} >/dev/null
            cp "$log" "$out/logs/forward-bf16-rules-${c%%:*}.log"
            if [ "$status" != 0 ] && grep -q "^FAIL .*: not forwarded for the ${c#*:}$" "$log"; then
                echo "PASS  bf16-rules control: without the ${c#*:} (${c%%:*}) it fails there"
            else echo "FAIL  bf16-rules control ${c%%:*}: did not fail at the ${c#*:} ($out/logs/forward-bf16-rules-${c%%:*}.log)"; failed=1; fi
        done
        run bf16-rules "$test_dir" VF_RULES_NO_BOUND=5 >/dev/null
        cp "$log" "$out/logs/forward-bf16-rules-no-bound.log"
        if [ "$status" = 1 ] && grep -q "^FAIL .*: the runtime reports no m bound of the tall rule's long-k branch" "$log"; then
            echo "PASS  bf16-rules control: a bound the runtime does not report fails its check (exit status 1, no crash)"
        else echo "FAIL  bf16-rules control VF_RULES_NO_BOUND=5: not that failed check with exit status 1 (exit status $status; $out/logs/forward-bf16-rules-no-bound.log)"; failed=1; fi
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
