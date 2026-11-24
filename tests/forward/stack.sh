#!/bin/bash
# Test that the status-renumbering trampolines of the forwarding libraries copy exactly the
# right number of stack words.  Needs no GPU.  Builds a mock NPP library, generates a forwarder
# that points at it (using the same vf-fwdgen the project uses), and runs the test harness.
#   tests/forward/stack.sh [out_dir]
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
out=${1:-${TMPDIR:-/tmp}/vf-fwd-stack-test}
. "$root/scripts/target-dir.sh"
CARGO_TARGET_DIR=$(vf_target_dir "$root")
export CARGO_TARGET_DIR
cargo=${CARGO:-$(command -v cargo || echo "$HOME/.cargo/bin/cargo")}

# The headers and libraries for the mock.
mock_hdr_new=$root/tests/forward/mock_npp_hdr     # "13.5" header (new enum values)
mock_hdr_old=$root/tests/forward/mock_npp_hdr_old # "12.9" header (old enum values)
mock_src=$root/tests/forward/mock_npp.c

mkdir -p "$out/new/lib" "$out/new/include" "$out/old/lib" "$out/old/include" "$out/forward" "$out/bin"

# The generator looks for headers at ../include relative to the lib directory, so each toolkit
# needs a lib/ and include/ side by side.
cp "$mock_hdr_new/npp.h" "$out/new/include/npp.h"
cp "$mock_hdr_old/npp.h" "$out/old/include/npp.h"

echo "building the mock native library..."
# Build the mock library twice: once as the "12.9" target, once as the "13.5" source for the ELF.
# Both export the same symbols; only the header enum values differ.
# The "12.9" lib is what the forwarder calls at runtime.
# The "13.5" lib is read by the generator for its export list.
# The generator reads the "13.5" library's exports and rejects data symbols, so mock_last
# is hidden from the version script of the new library.  The old library exports it normally
# so the test harness can read it via dlsym.
gcc -shared -fPIC -O2 -I"$mock_hdr_old" -Wl,-soname,libnppif.so.12 \
    -o "$out/old/lib/libnppif.so.12" "$mock_src" -Wl,--version-script=/dev/stdin <<'MAP'
{ global: nppi*; mock_last; local: *; };
MAP
gcc -shared -fPIC -O2 -I"$mock_hdr_new" -Wl,-soname,libnppif.so.13 \
    -o "$out/new/lib/libnppif.so.13" "$mock_src" -Wl,--version-script=/dev/stdin <<'MAP'
{ global: nppi*; local: *; };
MAP

echo "generating the forwarder library with vf-fwdgen..."
"$cargo" build --release -p vf-fwdgen 2>&1 | grep -v '^\s*Compiling\|Finished\|warning' || true
gen=$CARGO_TARGET_DIR/release/vf-fwdgen
"$gen" generate \
    --old "$out/old/lib" --new "$out/new/lib" --out "$out/forward" \
    --only nppif
"$gen" verify \
    --old "$out/old/lib" --new "$out/new/lib" --out "$out/forward" \
    --only nppif

echo "building the test harness..."
gcc -O2 -Wall -Wno-unused-result -Wno-unused-variable -I"$root/tests/forward" \
    -o "$out/bin/test_stack" \
    "$root/tests/forward/test_stack.c" \
    "$root/tests/forward/call_guarded.S" \
    -L"$out/old/lib" -Wl,-rpath,"$out/old/lib" -l:libnppif.so.12 \
    -ldl

echo "running the test..."
# Tell the forwarder where to find the "12.9" mock library at runtime.
VF_FWD_NPPIF="$out/old/lib/libnppif.so.12" \
    "$out/bin/test_stack" "$out/forward/libnppif.so.13"
