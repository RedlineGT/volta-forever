#!/bin/bash
# Single llama.cpp operations on a machine with a V100, official CUDA 13.5 build through the proxy
# against a build of the same sources for sm_70 (see tests/README.md for how to make it):
#   tests/gpu/llamacpp-ops.sh [GPU [MODE [OPERATION...]]]
# Runs llama.cpp's own backend test (tests/test-backend-ops.cpp of the b12418 sources), linked once
# against each set of libraries. MODE:
#   perf (default)  times every case of the operations; prints, per operation, how the time through the
#                   proxy compares with the native time, and leaves one line per case in
#                   $out/logs/ops-<operation>.txt (microseconds through the proxy, natively, ratio).
#   test            compares every case with the CPU backend; prints the counts of OK and FAIL. A
#                   program that stops before the end (a kernel refused, a fault) fails.
# OPERATION defaults to FLASH_ATTN_EXT and MUL_MAT. GPU is the PCI-order index (default 0). Timings are
# only meaningful on an idle GPU. The first run compiles every kernel variant the cases use (minutes).
# To compare two versions of the proxy: $VF_PROXY is the directory of the libcuda.so.1 to use (default:
# dist/lib of this tree) and $VF_TAG a suffix for the names of the log files; with $VF_REUSE_NATIVE
# set, the native timings of an earlier run are kept instead of being measured again.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
D=$W/demos/llamacpp/b12418
B=$D/llama/llama-b12418
T=$W/toolkits/cuda-12.9.2/lib
src=$W/src/llama.cpp-b12418
out=${VF_OUT:-$VF_TEST_ROOT}
native=${VF_NATIVE_LLAMA:-$VF_TEST_ROOT/llama-b12418-sm70/build/bin}
proxy_dir=${VF_PROXY:-$W/dist/lib} tag=${VF_TAG:-}
gpu=${1:-0} mode=${2:-perf}
[ $# -gt 2 ] && shift 2 || set --
[ $# -gt 0 ] || set -- FLASH_ATTN_EXT MUL_MAT
mkdir -p "$out/logs" "$out/tbo"
[ -e "$W/dist/forward/libcublas.so.13" ] || { echo "missing: $W/dist/forward (run scripts/build-forwarders.sh)"; exit 2; }
build() {  # build LIBRARIES PROGRAM : the backend test linked against the ggml libraries in LIBRARIES
    [ -x "$2" ] || g++ -O2 -std=c++17 "$src/tests/test-backend-ops.cpp" -I "$src/ggml/include" -I "$src/ggml/src" \
        -L "$1" -l:libggml.so.1 -l:libggml-base.so.1 -Wl,--allow-shlib-undefined -pthread -o "$2" || exit 2
}
build "$B" "$out/tbo/test-backend-ops.official"
[ -d "$native" ] && build "$native" "$out/tbo/test-backend-ops.native"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu
. "$W/tests/gpu/cache.sh"
export CUDA_CACHE_PATH=${VF_JIT_CACHE:-$VF_TEST_ROOT/jit-cache} CUDA_CACHE_MAXSIZE=4294967296
# The program looks for the backends in its working directory.
proxy() { (cd "$B" && LD_LIBRARY_PATH=$proxy_dir:$W/dist/forward:$B:$D/cudart/cudart-llama-b12418-bin-ubuntu-cuda-13.5-x64:$T \
    "$out/tbo/test-backend-ops.official" "$@"); }
natively() { (cd "$native" && LD_LIBRARY_PATH=$native:$T "$out/tbo/test-backend-ops.native" "$@"); }
echo "GPU $gpu before the runs (utilisation, memory used): $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i "$gpu")"
failed=0
for operation in "$@"; do
    log=$out/logs/ops-$operation$tag
    if [ "$mode" = test ]; then
        proxy test -o "$operation" -b CUDA0 >"$log-test-proxy.log" 2>&1
        status=$?
        # (The lines of the cases only, "  OPERATION(PARAMETERS): ... OK" or "... FAIL", the colours
        # taken away: not "  Backend CUDA0: OK", the last "OK", nor any other line with the word.)
        cases=$(sed 's/\x1b\[[0-9;]*m//g' "$log-test-proxy.log" | grep -a '^  [A-Za-z0-9_][A-Za-z0-9_]*(.*): .*\(OK\|FAIL\)$')
        ok=$(printf '%s\n' "$cases" | grep -c 'OK$') bad=$(printf '%s\n' "$cases" | grep -c 'FAIL$')
        if [ "$ok" -gt 0 ] && [ "$bad" = 0 ] && [ "$status" = 0 ]; then verdict=PASS; else verdict=FAIL; failed=1; fi
        stopped=
        [ "$status" = 0 ] || [ "$bad" != 0 ] || stopped=", then the program stopped (status $status): $(grep -a -m1 "cannot run on this GPU\|CUDA error" "$log-test-proxy.log" | cut -c1-200)"
        echo "$verdict  $operation through the proxy: $ok cases like the CPU backend, $bad not$stopped ($log-test-proxy.log)"
        continue
    fi
    native_log=$out/logs/ops-$operation-perf-native.log
    proxy perf -o "$operation" -b CUDA0 >"$log-perf-proxy.log" 2>&1
    if ! grep -q " us/run" "$log-perf-proxy.log"; then
        echo "FAIL  $operation: nothing was timed through the proxy:"; head -3 "$log-perf-proxy.log"; failed=1; continue
    fi
    if [ -x "$out/tbo/test-backend-ops.native" ] && { [ -z "${VF_REUSE_NATIVE:-}" ] || [ ! -s "$native_log" ]; }; then
        natively perf -o "$operation" -b CUDA0 >"$native_log" 2>&1
    fi
    # "  OPERATION(case): N runs - T us/run - ..." on both sides, joined by case.
    awk -v operation="$operation" '
        function parse(line) { time = line; sub(/^.*runs - +/, "", time); sub(/ us\/run.*$/, "", time)
                               sub(/^ +/, "", line); sub(/\): .*$/, ")", line); return line }
        index($0, "  " operation "(") == 1 && / us\/run/ {
            key = parse($0)
            if (FILENAME ~ /native\.log$/) { if (key in through) printf "%10.2f %10.2f %6.2f  %s\n", through[key], time, through[key] / time, key }
            else through[key] = time
        }' "$log-perf-proxy.log" "$native_log" 2>/dev/null >"$log.txt"
    cases=$(wc -l <"$log.txt")
    if [ "$cases" -gt 0 ]; then
        sort -n -k3 "$log.txt" | awk -v operation="$operation" -v file="$log.txt" '
            { ratio[NR] = $3; proxy += $1; native += $2 }
            END { printf "%s: %d cases, time through the proxy over native time: lowest %.2f, quartiles %.2f %.2f %.2f, highest %.2f; all cases together %.2f (%s)\n",
                  operation, NR, ratio[1], ratio[int((NR + 3) / 4)], ratio[int((NR + 1) / 2)], ratio[int((3 * NR + 1) / 4)], ratio[NR], proxy / native, file }'
    else
        echo "$operation: $(grep -c " us/run" "$log-perf-proxy.log") cases timed through the proxy ($log-perf-proxy.log), no native timing to compare with"
    fi
done
exit $failed
