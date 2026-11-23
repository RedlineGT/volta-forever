#!/bin/bash
# The refusal of a refused kernel at its use, under threads and in the orders that threads can take,
# without a GPU:
#   tests/proxy/guard_stress.sh [PROXY_DIR]
# PROXY_DIR is the directory of the libcuda.so.1 to test (default: dist/lib of this tree). A
# stand-in for the driver that threads can share (guard_mt_mock_driver.c) gives the handles of what
# is gone again, last freed first, to any thread; guard_stress.c says what each scenario does and
# checks. Each scenario runs in a process of its own, those with threads for
# VF_GUARD_STRESS_SECONDS seconds each (default 1; 60 or more for a long run) with
# VF_GUARD_STRESS_THREADS threads (default 8). Needs gcc.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
proxy=${1:-$W/dist/lib}
G=$W/tests/proxy
seconds=${VF_GUARD_STRESS_SECONDS:-1}
threads=${VF_GUARD_STRESS_THREADS:-8}
out=$(mktemp -d "${TMPDIR:-/tmp}/vf-guard-stress.XXXXXX") || exit 2
trap 'rm -rf "$out"' EXIT

gcc -shared -fPIC -O1 -Wl,-Bsymbolic -o "$out/mock.so" "$G/guard_mt_mock_driver.c" -lpthread || exit 2
gcc -O2 -Wall -o "$out/guard_stress" "$G/guard_stress.c" -ldl -lpthread || exit 2

export VF_REAL_LIBCUDA="$out/mock.so" VF_PROXY_CACHE=0 VF_LOG_LEVEL=0 VF_LAZY=0
status=0
for scenario in modules contexts green; do
    "$out/guard_stress" "$proxy/libcuda.so.1" "$out/mock.so" $scenario "$threads" "$seconds" || status=1
done
for scenario in samename nocontext; do
    "$out/guard_stress" "$proxy/libcuda.so.1" "$out/mock.so" $scenario || status=1
done

if [ $status -eq 0 ]; then echo "PASS  guard_stress"; else echo "FAIL  guard_stress"; fi
exit $status
