#!/bin/sh
# Make the archive used to test install.sh after scripts/build.sh. The 22 CUDA
# forwarders are omitted from this fixture. scripts/package.sh builds the release.
set -eu

if [ "$#" -ne 1 ] || [ -z "$1" ]; then
    echo "usage: $0 OUT_DIR" >&2
    exit 2
fi
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
mkdir -p -- "$1"
out=$(CDPATH= cd -- "$1" && pwd)
version=$(sed -n 's/^version = "\(.*\)"$/\1/p' "$root/Cargo.toml" | head -1)
[ -n "$version" ] || { echo "no version in Cargo.toml" >&2; exit 1; }
commit=$(git -C "$root" rev-parse --short HEAD)
name=volta-forever-$version-linux-x86_64

stage=$(mktemp -d "$out/.installer-fixture.XXXXXXXX")
trap 'rm -rf -- "$stage"' EXIT
pkg=$stage/$name
mkdir -p "$pkg/bin" "$pkg/lib" "$pkg/scripts"
for file in bin/vf-translate bin/vf-ptxas lib/libcuda.so.1 lib/libnvJitLink.so.13 lib/libvf_audit.so; do
    if [ ! -s "$root/dist/$file" ]; then
        echo "missing or empty dist/$file (run scripts/build.sh first)" >&2
        exit 1
    fi
    cp "$root/dist/$file" "$pkg/$file"
done
cp "$root/scripts/vf" "$pkg/bin/vf"
ln -s vf "$pkg/bin/volta-forever"
cp "$root/scripts/fetch-redist.py" "$pkg/scripts/"
printf '%s %s\n' "$version" "$commit" > "$pkg/VERSION"

cd "$stage"
tar -czf "$name.tar.gz" "$name"
sha256sum "$name.tar.gz" > SHA256SUMS
mv "$name.tar.gz" SHA256SUMS "$out/"
