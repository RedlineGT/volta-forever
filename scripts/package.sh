#!/bin/bash
# Build a distributable archive of volta-forever from a built tree.
#   scripts/package.sh [VERSION]
#
# The archive holds what a user needs to run CUDA 13 programs on a V100 and nothing of NVIDIA's:
#   lib/            the proxy (libcuda.so.1), the audit library, the nvJitLink adapter
#   forward/        forwarding libraries: CUDA 13 names served by CUDA 12.9 (generated, no NVIDIA code)
#   forward/bf16tc/ the bf16 cuBLAS variants (libcublas.so.13, libcublasLt.so.13) and their runtime
#                   library libvf_cublas_bf16.so, when they were built
#   bin/vf            the launcher (sets the environment and runs a program), with a link
#                     bin/volta-forever to it
#   bin/vf-translate  the offline tool (inspect, extract, adapt, translate)
#   scripts/        fetch-redist.py (to download the CUDA 12.9 libraries the user needs)
#   tuning/         measured kernel configurations in the format read by each engine
#   INSTALL.md      installation and usage
#   LIMITS.md       known limits and measurements
#   docs/           how the layer works, what it computes, how it is checked, what it costs
#   LICENSE         the licence (Apache-2.0)
#   LICENSE.LGPL-2.1  the licence of the instruction tables that the binaries contain
#   NOTICE          what the binaries contain that is not this project's, and where their source is
#   VERSION         version, commit, dirty flag
#
# Refuses to run on a dirty tree (tracked files differ from HEAD) unless ALLOW_DIRTY=1.
# Refuses to run when the built tree lacks one of the forwarding libraries that a whole run of
# scripts/build-forwarders.sh makes (`vf-fwdgen list`), when it holds some but not all of the three
# files of the bf16 cuBLAS variants, or when a binary or a forwarding library is older than the
# sources.
# $DIST: the built tree to pack and where the archive goes (default: dist/ of this tree).
# Prints what it packed and the checksums.
# Output: volta-forever-<version>-linux-x86_64.tar.gz (what install.sh downloads), the same as
# .tar.zst when zstd is there, and SHA256SUMS, which names them.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
# (The version of the workspace, unless another is given.)
version=${1:-$(sed -n 's/^version = "\(.*\)"$/\1/p' "$root/Cargo.toml" | head -1)}
[ -n "$version" ] || { echo "error: no version given and none found in Cargo.toml"; exit 2; }

# ---- check dirty tree ----------------------------------------------------------------------------------
commit=$(cd "$root" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
dirty=""
if ! (cd "$root" && git diff --quiet HEAD -- 2>/dev/null); then
    dirty="-dirty"
    if [ "${ALLOW_DIRTY:-0}" != 1 ]; then
        echo "error: tracked files differ from HEAD (commit or set ALLOW_DIRTY=1)"
        exit 2
    fi
fi
version_string="$version $commit$dirty"

out=${DIST:-$root/dist}

# ---- check that the tree is built and not stale -------------------------------------------------------
bins=("$out/lib/libcuda.so.1" "$out/lib/libnvJitLink.so.13" "$out/lib/libvf_audit.so" "$out/bin/vf-translate" "$out/bin/vf-ptxas")
for f in "${bins[@]}"; do
    [ -f "$f" ] || { echo "error: $f not found (run scripts/build.sh first)"; exit 2; }
done
# The forwarding libraries: every one that the generator makes in a whole run, built, with its
# generated source beside it and its line in the report. (A run for some of them only,
# `build-forwarders.sh DIR cublas`, leaves the others as they were or absent.)
. "$root/scripts/target-dir.sh"
gen=$(vf_target_dir "$root")/release/vf-fwdgen
[ -x "$gen" ] || { echo "error: $gen not found (run scripts/build-forwarders.sh first)"; exit 2; }
mapfile -t forwarders < <("$gen" list)
[ "${#forwarders[@]}" -gt 0 ] || { echo "error: $gen lists no forwarding library (run scripts/build-forwarders.sh first)"; exit 2; }
missing=()
for lib in "${forwarders[@]}"; do
    [ -f "$out/forward/$lib" ] && [ -d "$out/forward/gen/$lib" ] && grep -q "^$lib " "$out/forward/REPORT.txt" 2>/dev/null || missing+=("$lib")
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "error: $out/forward lacks ${#missing[@]} of the ${#forwarders[@]} forwarding libraries, or its report does not list them: ${missing[*]} (run scripts/build-forwarders.sh with no selection)"
    exit 2
fi
for lib in "${forwarders[@]}"; do
    bins+=("$out/forward/$lib")
done
# The bf16 cuBLAS variants (forward/bf16tc), when they were built: the three files, or none. The
# launcher refuses VF_CUBLAS_BF16_TC=1 and VF_CUBLAS_BF16_SKINNY=1 unless all three are there.
bf16_files=(libcublas.so.13 libcublasLt.so.13 libvf_cublas_bf16.so)
bf16_there=()
bf16_missing=()
for lib in "${bf16_files[@]}"; do
    if [ -f "$out/forward/bf16tc/$lib" ]; then bf16_there+=("$lib"); else bf16_missing+=("$lib"); fi
done
if [ "${#bf16_there[@]}" -gt 0 ] && [ "${#bf16_missing[@]}" -gt 0 ]; then
    echo "error: $out/forward/bf16tc holds ${#bf16_there[@]} of the ${#bf16_files[@]} files of the bf16 cuBLAS variants: it lacks ${bf16_missing[*]} (build all three, or remove the directory)"
    exit 2
fi
if [ "${#bf16_missing[@]}" -eq 0 ]; then
    for lib in "${bf16_files[@]}"; do
        bins+=("$out/forward/bf16tc/$lib")
    done
fi

# Check staleness: the oldest binary or forwarding library must be newer than the sources.
oldest_bin="${bins[0]}"
for f in "${bins[@]}"; do
    [ "$f" -ot "$oldest_bin" ] && oldest_bin="$f"
done
for src in "$root/crates" "$root/Cargo.toml" "$root/Cargo.lock" "$root/scripts/build.sh" "$root/scripts/build-forwarders.sh" "$root/scripts/target-dir.sh"; do
    [ -e "$src" ] || continue
    newest_src=$(find "$src" -newer "$oldest_bin" -print -quit 2>/dev/null || true)
    if [ -n "$newest_src" ]; then
        echo "error: $(basename "$oldest_bin") is older than $newest_src (rebuild first)"
        exit 2
    fi
done

# ---- assemble the archive in a temporary directory -----------------------------------------------------
name="volta-forever-${version}-linux-x86_64"
staging=$(mktemp -d "${TMPDIR:-/tmp}/volta-forever-pkg.XXXXXXXX")
trap 'rm -rf "$staging"' EXIT
pkg="$staging/$name"
mkdir -p "$pkg/lib" "$pkg/forward" "$pkg/bin" "$pkg/scripts" "$pkg/python"

# lib/: the proxy, the audit library, the nvJitLink adapter
cp "$out/lib/libcuda.so.1"       "$pkg/lib/"
cp "$out/lib/libnvJitLink.so.13" "$pkg/lib/"
cp "$out/lib/libvf_audit.so"   "$pkg/lib/"

# bin/: the launcher under its two names, and the offline tool
cp "$out/bin/vf-translate" "$out/bin/vf-ptxas" "$pkg/bin/"
cp "$root/scripts/vf" "$pkg/bin/vf"
chmod +x "$pkg/bin/vf"
ln -s vf "$pkg/bin/volta-forever"

# forward/: the forwarding libraries of the list, and the report of the run that made them
for lib in "${forwarders[@]}"; do
    cp "$out/forward/$lib" "$pkg/forward/"
done
cp "$out/forward/REPORT.txt" "$pkg/forward/"
# forward/bf16tc/: the bf16 cuBLAS variants and their runtime library, when they were built
if [ "${#bf16_missing[@]}" -eq 0 ]; then
    mkdir -p "$pkg/forward/bf16tc"
    for lib in "${bf16_files[@]}"; do
        cp "$out/forward/bf16tc/$lib" "$pkg/forward/bf16tc/"
    done
fi

# python/: what the launcher puts first in PYTHONPATH with VF_TRITON_TC (Triton's products)
cp "$root/python/sitecustomize.py" "$root/python/volta_forever_triton.py" \
    "$root/python/volta_forever_triton_client.py" "$root/python/volta_forever_triton_helper.py" "$pkg/python/"
# An empty file where Python would make python/__pycache__: it then writes no compiled file into
# the installation when a program starts (the modules of VF_TRITON_TC are read without writing
# one).
: >"$pkg/python/__pycache__"

# scripts/: what a user needs to fetch the CUDA 12.9 libraries
cp "$root/scripts/fetch-redist.py" "$pkg/scripts/"
chmod +x "$pkg/scripts/fetch-redist.py"

# Keep the directory structure expected by the engines when reading their tuning data.
(cd "$root" && find tuning -type f -name '*.json' -exec cp --parents -t "$pkg" {} +)

# Documentation
for doc in INSTALL.md LIMITS.md; do
    [ -f "$root/$doc" ] && cp "$root/$doc" "$pkg/"
done
echo "$version_string" > "$pkg/VERSION"

if [ -d "$root/docs" ]; then
    mkdir -p "$pkg/docs" && cp "$root"/docs/*.md "$pkg/docs/"
fi

# The licences, and the notice of what the binaries contain that is not this project's
for f in "$root"/LICENSE* "$root"/LICENCE* "$root"/COPYING* "$root"/NOTICE*; do
    [ -f "$f" ] && cp "$f" "$pkg/"
done
# The instruction tables in the binaries have a licence of their own (NOTICE, section 1).
cp "$root/crates/vf-tables/LICENSE" "$pkg/LICENSE.LGPL-2.1"

# ---- verify: only expected files, and forwarding libraries need libc only ------------------------------
echo "--- verifying archive contents ---"
expected_lib="libcuda.so.1 libnvJitLink.so.13 libvf_audit.so"
expected_bin="vf vf-ptxas vf-translate volta-forever"
expected_python="__pycache__ sitecustomize.py volta_forever_triton.py volta_forever_triton_client.py volta_forever_triton_helper.py"
fail=0

# Check lib/: only the three expected files
for f in "$pkg/lib/"*; do
    [ -f "$f" ] || continue
    fname=$(basename "$f")
    case " $expected_lib " in
        *" $fname "*) ;;
        *) echo "ERROR: unexpected file in lib/: $fname"; fail=1 ;;
    esac
done

# Check bin/: only vf
for f in "$pkg/bin/"*; do
    [ -f "$f" ] || continue
    fname=$(basename "$f")
    case " $expected_bin " in
        *" $fname "*) ;;
        *) echo "ERROR: unexpected file in bin/: $fname"; fail=1 ;;
    esac
done

# Check python/: its four modules and the empty file __pycache__ alone
for f in "$pkg/python/"*; do
    fname=$(basename "$f")
    case " $expected_python " in
        *" $fname "*) ;;
        *) echo "ERROR: unexpected file in python/: $fname"; fail=1 ;;
    esac
done

# Check forward/: each .so must have a gen/ directory in the build tree, and be one of the list
for f in "$pkg/forward/"*.so.*; do
    [ -f "$f" ] || continue
    fname=$(basename "$f")
    if [ ! -d "$out/forward/gen/$fname" ]; then
        echo "ERROR: $fname has no generated source in forward/gen/: may be an NVIDIA file"
        fail=1
    fi
    case " ${forwarders[*]} " in
        *" $fname "*) ;;
        *) echo "ERROR: unexpected file in forward/: $fname"; fail=1 ;;
    esac
done
if [ "$(find "$pkg/forward" -maxdepth 1 -name '*.so.*' -type f | wc -l)" != "${#forwarders[@]}" ]; then
    echo "ERROR: forward/ does not hold the ${#forwarders[@]} forwarding libraries"
    fail=1
fi

# Check forward/bf16tc/: the three files of the bf16 cuBLAS variants and nothing else, or nothing
if [ -e "$pkg/forward/bf16tc" ]; then
    for f in "$pkg/forward/bf16tc/"*; do
        fname=$(basename "$f")
        case " ${bf16_files[*]} " in
            *" $fname "*) ;;
            *) echo "ERROR: unexpected file in forward/bf16tc/: $fname"; fail=1 ;;
        esac
    done
    if [ "$(find "$pkg/forward/bf16tc" -type f | wc -l)" != "${#bf16_files[@]}" ]; then
        echo "ERROR: forward/bf16tc/ does not hold the ${#bf16_files[@]} files of the bf16 cuBLAS variants"
        fail=1
    fi
    # The runtime: its exports, the libraries it needs, and the kernels of kernels.manifest in it
    if ! "$gen" check-bf16 "$pkg/forward/bf16tc"; then
        echo "ERROR: forward/bf16tc/libvf_cublas_bf16.so does not pass vf-fwdgen check-bf16"
        fail=1
    fi
fi

# Forwarding libraries (the bf16 variants included) must need only libc.so.6
for f in "$pkg/forward/"*.so.* "$pkg/forward/bf16tc/"libcublas*.so.*; do
    [ -f "$f" ] || continue
    extra=$(readelf -d "$f" 2>/dev/null | grep NEEDED | grep -v "libc.so.6" | head -1 || true)
    if [ -n "$extra" ]; then
        echo "ERROR: $(basename "$f") needs more than libc: $extra"
        fail=1
    fi
done

if [ "$fail" -eq 1 ]; then
    echo "error: verification failed; aborting"
    exit 3
fi

# ---- build the archive ---------------------------------------------------------------------------------
# The same tree gives the same archive, whoever packs it, where and when: the files in the order
# of their names, their times the time of the commit ($SOURCE_DATE_EPOCH when it is set), their
# modes not what the packer's umask made them, no name of a user, and a compression that does
# not depend on how many cores the machine has.
epoch=${SOURCE_DATE_EPOCH:-$(cd "$root" && git log -1 --format=%ct 2>/dev/null || true)}
[ -n "$epoch" ] || epoch=$(date +%s)
pack() {
    tar -C "$staging" --format=ustar --sort=name --mtime="@$epoch" --owner=0 --group=0 \
        --numeric-owner --mode='u=rwX,go=rX' -cf - "$name"
}
echo "--- packing $name ---"
# A .tar.gz always: it is what install.sh downloads, and gzip is on every system. A .tar.zst
# too when zstd is there (smaller). SHA256SUMS names what was written, for whoever downloads.
archives=("${name}.tar.gz")
pack | gzip -n -9 > "$staging/${name}.tar.gz"
if command -v zstd >/dev/null 2>&1; then
    archives+=("${name}.tar.zst")
    pack | zstd -9 --single-thread > "$staging/${name}.tar.zst"
fi
for archive in "${archives[@]}"; do
    cp "$staging/$archive" "$out/$archive"
done
(cd "$out" && sha256sum "${archives[@]}" > SHA256SUMS)
# (What is listed below: the last one written, the .tar.zst where there is one.)
archive=${archives[-1]}
echo ""
echo "=== archive: $out/$archive ==="
ls -lh "$out/$archive"
echo ""

# ---- list what it holds and print checksums ------------------------------------------------------------
echo "=== contents ==="
if [[ "$archive" == *.zst ]]; then
    zstd -d --stdout "$out/$archive" | tar -tf - | sort
else
    tar -tf "$out/$archive" | sort
fi
echo ""
echo "=== sizes ==="
du -sh "$pkg/lib" "$pkg/forward" "$pkg/bin" "$pkg/scripts"
echo ""
echo "=== checksums ($out/SHA256SUMS) ==="
cat "$out/SHA256SUMS"
echo ""
echo "--- file checksums ---"
find "$pkg" -type f | sort | while read -r f; do
    sha256sum "$f" | sed "s|$staging/||"
done
echo ""
echo "done: $out/$archive (version $version_string)"
