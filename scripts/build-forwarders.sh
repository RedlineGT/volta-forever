#!/bin/bash
# Build the forwarding libraries into dist/forward/ (or the directory given): for each CUDA 13 library
# a program asks for (cuBLAS, cuBLASLt, cuFFT, cuFFTW, cuSOLVER, cuSOLVERMg, cuSPARSE, cuRAND, the NPP
# libraries, nvJPEG, NVRTC and its builtins), a library with the same name, the same exports and the
# same symbol versions, whose functions call those of CUDA 12.9.2 (the first call of each loads it).
# Then checks them on the host. Reads toolkits/cuda-13.5.0/ and toolkits/cuda-12.9.2/ (lib and include);
# needs gcc. Writes dist/forward/REPORT.txt, which says what was generated.
#   scripts/build-forwarders.sh [OUT_DIR [KEY,KEY...]]      e.g. scripts/build-forwarders.sh dist/forward cublas,cufft
# Same inputs, same bytes: running it twice gives identical files.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
. "$root/scripts/target-dir.sh"
CARGO_TARGET_DIR=$(vf_target_dir "$root")
export CARGO_TARGET_DIR
cargo=${CARGO:-$(command -v cargo || echo "$HOME/.cargo/bin/cargo")}
out=${1:-$root/dist/forward}
only=${2:+--only $2}
cd "$root"
"$cargo" build --release -p vf-fwdgen
gen=$CARGO_TARGET_DIR/release/vf-fwdgen
# shellcheck disable=SC2086
"$gen" generate --old toolkits/cuda-12.9.2/lib --new toolkits/cuda-13.5.0/lib --out "$out" $only
# shellcheck disable=SC2086
"$gen" verify --old toolkits/cuda-12.9.2/lib --new toolkits/cuda-13.5.0/lib --out "$out" $only
echo "forwarders in $out (report: $out/REPORT.txt)"
