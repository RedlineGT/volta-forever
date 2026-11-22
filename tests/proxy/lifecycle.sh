#!/bin/bash
# What the layer keeps over the life of a process, and what it does at its end, without a GPU:
#   tests/proxy/lifecycle.sh [PROXY_DIR]
# PROXY_DIR is the directory of the libcuda.so.1 to test (default: dist/lib of this tree). A
# stand-in for the driver (lifecycle_mock_driver.c) gives every module a handle of its own;
# lifecycle.c says what each case checks. Each case runs in a process of its own. Needs gcc.
# Prints one line a check, then PASS or FAIL; exit status 1 on FAIL.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
proxy=${1:-$W/dist/lib}
G=$W/tests/proxy
out=$(mktemp -d "${TMPDIR:-/tmp}/vf-lifecycle-test.XXXXXX") || exit 2
trap 'rm -rf "$out"' EXIT

gcc -shared -fPIC -O0 -Wl,-Bsymbolic -o "$out/mock.so" "$G/lifecycle_mock_driver.c" || exit 2
gcc -O2 -Wall -o "$out/lifecycle" "$G/lifecycle.c" -ldl -lpthread || exit 2

# The stand-in as the driver; nothing kept, nothing said unless a case asks for it.
export VF_REAL_LIBCUDA="$out/mock.so"
export VF_PROXY_CACHE=0
export VF_LOG_LEVEL=0
export VF_EMULATION_WARNINGS=0
status=0
for case in fork_exit; do
    mkdir -p "$out/$case"
    "$out/lifecycle" "$case" "$proxy/libcuda.so.1" "$out/$case" || status=1
done

if [ $status -eq 0 ]; then echo "PASS  lifecycle"; else echo "FAIL  lifecycle"; fi
exit $status
