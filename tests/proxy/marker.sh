#!/bin/bash
# The functions by which a program can tell that the layer is there, without a GPU:
#   tests/proxy/marker.sh [PROXY_DIR]
# PROXY_DIR is the directory of the libcuda.so.1 to test (default: dist/lib of this tree). The
# version expected is the workspace's (Cargo.toml). A stand-in for the driver is given to the
# layer (fit_mock_driver.c): the functions call nothing of it. Also that the library names no path
# of the crates as the build host keeps them (scripts/build.sh remaps them to cargo/...; until
# 2026-11-25 it remapped ~/.cargo alone, and a build under another HOME kept CARGO_HOME's paths and
# gave other bytes). Needs gcc.
# Prints one line a check, then PASS or FAIL; exit status 1 on FAIL.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
proxy=${1:-$W/dist/lib}
G=$W/tests/proxy
out=$(mktemp -d "${TMPDIR:-/tmp}/vf-marker-test.XXXXXX") || exit 2
trap 'rm -rf "$out"' EXIT

gcc -shared -fPIC -O0 -Wl,-Bsymbolic -o "$out/mock.so" "$G/fit_mock_driver.c" || exit 2
gcc -O2 -Wall -o "$out/marker" "$G/marker.c" -ldl || exit 2
version=$(sed -n 's/^version = "\(.*\)"$/\1/p' "$W/Cargo.toml" | head -1)

export VF_REAL_LIBCUDA="$out/mock.so"
export VF_PROXY_CACHE=0
export VF_LOG_LEVEL=0
"$out/marker" "$proxy/libcuda.so.1" "$version"
status=$?
for crates in /.cargo/registry/ ${CARGO_HOME:+"$CARGO_HOME/registry/"}; do
    if grep -q -a -F -- "$crates" "$proxy/libcuda.so.1"; then
        echo "FAIL  the library names the build host's crates ($crates)"
        status=1
    else
        echo "ok    no path of the build host's crates ($crates)"
    fi
done

if [ $status -eq 0 ]; then echo "PASS  marker"; else echo "FAIL  marker"; fi
exit $status
