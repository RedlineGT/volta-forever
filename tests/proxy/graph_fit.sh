#!/bin/bash
# Offline counterfactual for the register-limited graph paths. The driver derives the
# resource claim from the .text section of the actual translated public test cubin.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
r=$(cd "$here/../.." && pwd)
proxy=${1:-$r/dist/lib/libcuda.so.1}
fixture=${2:-$r/tests/kernels/k_fit_test.135.sm75.cubin}
d=$(mktemp -d "${TMPDIR:-/tmp}/vf-graph-fit-test.XXXXXXXX")
trap 'rm -rf "$d"' EXIT
cc -std=c11 -fPIC -shared -O0 -Wl,-Bsymbolic "$here/graph_fit_driver.c" -o "$d/driver.so"
cc -std=c11 -O0 "$here/graph_fit.c" -ldl -o "$d/run"
export VF_REAL_LIBCUDA="$d/driver.so" VF_PROXY_CACHE=0 VF_LOG_LEVEL=0 VF_EMULATION_WARNINGS=0
for borrow in 7 5 9; do
  export VF_BORROW=$borrow
  for mode in small v1 v2 generic set update capture warmcapture; do
    expected=0
    if [[ $borrow == 7 && $mode != small ]]; then expected=701; fi
    "$d/run" "$proxy" "$fixture" "$mode" "$expected"
  done
done
echo "PASS  graph_fit"
