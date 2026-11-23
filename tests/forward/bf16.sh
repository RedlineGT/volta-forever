#!/bin/bash
# Host tests of the bf16 cuBLAS variants and their runtime (review of the design, section 5: H1-H7,
# H9), without a GPU and without NVIDIA's files: stand-ins for libcuda.so.1, libcublas.so.12 and
# libcublasLt.so.12 (tests/forward/bf16/stub_*.c) that record every call; mock "CUDA 13" libraries and
# headers (the headers are the runtime's own declarations, crates/vf-fwdgen/bf16tc/decls.h); the
# variants generated from them by vf-fwdgen, and checked by its verify; a test build of the runtime
# (-DVF_BF16_TEST) beside copies of them; tests/forward/bf16/host_test.c, one mode per process. Then
# each control (a mechanism broken on purpose) must make its test fail.
#   tests/forward/bf16.sh [OUT_DIR]
set -uo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
out=${1:-${TMPDIR:-/tmp}/vf-fwd-bf16-test}
src=$root/tests/forward/bf16
rt=$root/crates/vf-fwdgen/bf16tc
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/volta-forever/target}
cargo=${CARGO:-$(command -v cargo || echo "$HOME/.cargo/bin/cargo")}
rm -rf "$out"
mkdir -p "$out/old/lib" "$out/old/include" "$out/new/lib" "$out/new/include" "$out/test" "$out/noruntime" "$out/abort"
cc=(gcc -O2 -g0 -std=gnu11 -Wall -Wextra -Wno-unused-parameter -Werror -fPIC -I"$src")

echo "building the stand-ins and the mock libraries..."
"${cc[@]}" -shared -fvisibility=hidden -Wl,-soname,libcuda.so.1 -o "$out/old/lib/libcuda.so.1" "$src/stub_cuda.c" -lpthread || exit 1
for v in 12 13; do
    dir=$out/old/lib; [ $v = 13 ] && dir=$out/new/lib
    "${cc[@]}" -shared -fvisibility=hidden -Wl,-soname,libcublas.so.$v -o "$dir/libcublas.so.$v" "$src/stub_cublas.c" \
        -L"$out/old/lib" -l:libcuda.so.1 || exit 1
    "${cc[@]}" -shared -fvisibility=hidden -Wl,-soname,libcublasLt.so.$v -o "$dir/libcublasLt.so.$v" "$src/stub_cublaslt.c" \
        -L"$out/old/lib" -l:libcuda.so.1 || exit 1
done
for d in old new; do
    for h in cuda.h cublas_v2.h cublasXt.h cublas.h cublasLt.h library_types.h; do
        echo "#include \"$rt/decls.h\"" > "$out/$d/include/$h"
    done
done

echo "generating the variants with vf-fwdgen..."
"$cargo" build --release -p vf-fwdgen 2>&1 | grep -v '^\s*Compiling\|Finished\|warning' || true
gen=$CARGO_TARGET_DIR/release/vf-fwdgen
failed=0
"$gen" generate --old "$out/old/lib" --new "$out/new/lib" --out "$out/forward" --only cublas,cublasLt >"$out/generate.log" 2>&1 ||
    { echo "FAIL  vf-fwdgen generate"; cat "$out/generate.log"; exit 1; }
if "$gen" verify --old "$out/old/lib" --new "$out/new/lib" --out "$out/forward" --only cublas,cublasLt >"$out/verify.log" 2>&1; then
    echo "PASS  verify: the variants and the runtime ($(grep -c '^PASS' "$out/verify.log") files)"
else
    echo "FAIL  verify"; cat "$out/verify.log"; failed=1
fi

echo "building the test runtime and the test program..."
cp "$out/forward/bf16tc/libcublas.so.13" "$out/forward/bf16tc/libcublasLt.so.13" "$out/test/"
cp "$out/forward/bf16tc/libcublas.so.13" "$out/forward/bf16tc/libcublasLt.so.13" "$out/noruntime/"
"${cc[@]}" -shared -fvisibility=hidden -DVF_BF16_TEST -Wa,-I"$rt" -Wl,-soname,libvf_cublas_bf16.so \
    -o "$out/test/libvf_cublas_bf16.so" "$rt/runtime.c" "$rt/kernels.S" -Wl,-z,defs -ldl -lpthread || exit 1
"${cc[@]}" -o "$out/host_test" "$src/host_test.c" -L"$out/old/lib" -Wl,-rpath,"$out/old/lib" -l:libcuda.so.1 \
    -l:libcublas.so.12 -ldl || exit 1
# The variant built as the generator builds it, with the control that aborts when the runtime is missing.
g=$out/forward/gen/bf16tc/libcublas.so.13
gcc -shared -fPIC -O2 -fvisibility=hidden -DVF_BF16_CONTROL_ABORT -I"$out/new/include" -o "$out/abort/libcublas.so.13" \
    "$g/fwd.S" "$g/fwd.c" "$g/status.c" -Wl,-soname,libcublas.so.13 -Wl,--version-script="$g/ver.map" -ldl || exit 1
cp "$out/forward/bf16tc/libcublasLt.so.13" "$out/abort/"

run() {  # run NAME DIR ENV... -- ARGS... : the test program with LD_LIBRARY_PATH on the stand-ins
    local name=$1 dir=$2; shift 2
    local envs=()
    while [ "$1" != -- ]; do envs+=("$1"); shift; done
    shift
    log=$out/$name.log
    env -u VF_CUBLAS_BF16_TC -u VF_CUBLAS_BF16_SKINNY -u VF_BF16_CONTROL -u VF_BF16_INJECT -u STUB_CUBLAS_VERSION \
        LD_LIBRARY_PATH="$out/old/lib" "${envs[@]}" "$out/host_test" "$dir" "$@" >"$log" 2>&1
    status=$?
}
pass() {  # pass NAME WHAT : PASS when the program printed "ok" and exited with 0
    if [ "$status" = 0 ] && grep -q '^ok ' "$log"; then echo "PASS  $2"; else echo "FAIL  $2 (status $status, $log)"; failed=1; fi
}
fails() {  # fails NAME WHAT : PASS when the program failed (a control)
    if [ "$status" != 0 ]; then echo "PASS  $2"; else echo "FAIL  $2: the test passed with the control (${log})"; failed=1; fi
}

run all "$out/test" VF_CUBLAS_BF16_TC=1 -- all
pass all "H1-H7: forwarding with the same arguments, call sequences and chunks, explicit algorithms, bounds, one runtime, the internal GEMM, captured calls, the cuBLASLt allowlist"
run skinny "$out/test" VF_CUBLAS_BF16_SKINNY=1 -- skinny
pass skinny "H3: the skinny path's bounds and its rule for 12.9's sequential kernel"
run otherbuild "$out/test" VF_CUBLAS_BF16_SKINNY=1 STUB_CUBLAS_VERSION=120901 -- otherbuild
pass otherbuild "the skinny path with another cuBLAS than the 12.9.2 its rule was measured on: everything forwarded"
run off "$out/test" -- off
pass off "H1: no switch, everything forwarded and the runtime not loaded"
for step in module alloc scale gemm16 epilogue fallback; do
    run inject-$step "$out/test" VF_CUBLAS_BF16_TC=1 -- inject $step
    pass inject-$step "H1: a failure injected at $step"
done
run noruntime "$out/noruntime" VF_CUBLAS_BF16_TC=1 -- noruntime
said=$(grep -c "libcublas.so.13: cannot load .*libvf_cublas_bf16.so" "$log")
if [ "$status" = 0 ] && grep -q '^ok ' "$log" && [ "$said" = 1 ]; then
    echo "PASS  H9: the runtime missing: everything forwarded, said once, no abort"
else
    echo "FAIL  H9: status $status, said $said times ($log)"; failed=1
fi

echo "controls (each must make its test fail):"
for c in algo tc-k split-states caller-handle-any capture-tc lt-drop; do
    run control-$c "$out/test" VF_CUBLAS_BF16_TC=1 VF_BF16_CONTROL=$c -- all
    fails control-$c "control $c"
done
for c in skinny-k skinny-speed seq-any; do
    run control-$c "$out/test" VF_CUBLAS_BF16_SKINNY=1 VF_BF16_CONTROL=$c -- skinny
    fails control-$c "control $c"
done
run control-any-build "$out/test" VF_CUBLAS_BF16_SKINNY=1 STUB_CUBLAS_VERSION=120901 VF_BF16_CONTROL=any-build -- otherbuild
fails control-any-build "control any-build"
run control-abort "$out/abort" VF_CUBLAS_BF16_TC=1 -- noruntime
fails control-abort "control: abort when the runtime is missing"
exit $failed
