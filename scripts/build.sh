#!/bin/bash
# Build the Rust workspace and install the results under dist/:
#   dist/lib/libcuda.so.1   the proxy; put this directory first in LD_LIBRARY_PATH
#   dist/bin/vf-translate   the offline tool (inspect, extract, adapt, translate); the launcher
#                           scripts/vf has its subcommands and finds it there
# Build products go to $CARGO_TARGET_DIR, else to a directory of this checkout's own outside the
# tree, ~/.cache/volta-forever/target-HASH (scripts/target-dir.sh says why one for each checkout).
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
. "$root/scripts/target-dir.sh"
CARGO_TARGET_DIR=$(vf_target_dir "$root")
export CARGO_TARGET_DIR
cargo=${CARGO:-$(command -v cargo || echo "$HOME/.cargo/bin/cargo")}
cd "$root"
# No path of the build host in the binaries (panic messages, line tables). The crates are where cargo
# keeps them, under CARGO_HOME, which is not always $HOME/.cargo.
export RUSTFLAGS="${RUSTFLAGS:-} --remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=cargo --remap-path-prefix=$CARGO_TARGET_DIR=target --remap-path-prefix=$root=volta-forever"
"$cargo" build --release "$@"
mkdir -p dist/lib dist/bin
# Replace by rename: a node may have the previous file mapped.
install -m 755 "$CARGO_TARGET_DIR/release/libcuda.so" dist/lib/.libcuda.so.1.new
mv -f dist/lib/.libcuda.so.1.new dist/lib/libcuda.so.1
for tool in vf-translate vf-ptxas; do
    install -m 755 "$CARGO_TARGET_DIR/release/$tool" "dist/bin/.$tool.new"
    mv -f "dist/bin/.$tool.new" "dist/bin/$tool"
done
# libnvJitLink.so.13 served by nvJitLink 12.9.2, and the audit library that substitutes libraries.
install -m 755 "$CARGO_TARGET_DIR/release/libnvJitLink.so" dist/lib/.libnvJitLink.so.13.new
mv -f dist/lib/.libnvJitLink.so.13.new dist/lib/libnvJitLink.so.13
install -m 755 "$CARGO_TARGET_DIR/release/libvf_audit.so" dist/lib/.libvf_audit.so.new
mv -f dist/lib/.libvf_audit.so.new dist/lib/libvf_audit.so
echo "installed dist/lib/libcuda.so.1, libnvJitLink.so.13, libvf_audit.so and the VF compiler tools under dist/bin"
