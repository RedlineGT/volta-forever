#!/bin/bash
# What scripts/package.sh refuses to pack, without a GPU and without touching dist/:
#   tests/forward/package-refusals.sh
# Each case packs a copy of this tree's dist/ (built with scripts/build.sh and
# scripts/build-forwarders.sh), in a temporary directory, with one thing wrong:
#   - a forwarding library is missing (libcublasLt.so.13): refused, and named;
#   - a forwarding library is older than the sources: refused, and named;
#   - the report is that of a run for one library only: refused;
#   - forward/bf16tc holds some but not all of the three files of the bf16 cuBLAS variants (a
#     variant missing, the runtime library missing, the runtime library alone): refused, and
#     what is missing named.
# An archive of such a tree would not do what INSTALL.md says, or would carry libraries of an
# earlier generator under a new version. And one case with nothing wrong: forward/bf16tc with the
# three files built by scripts/build-forwarders.sh is packed (package.sh reads them: vf-fwdgen
# check-bf16 checks the runtime's exports and kernels), and the archive has them in forward/bf16tc,
# and nothing else there.
# Prints PASS or FAIL for each case; exit status 1 on a FAIL.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$W/dist/forward/libcublasLt.so.13" ] && [ -f "$W/dist/lib/libcuda.so.1" ] || { echo "missing: a built dist/ (scripts/build.sh, scripts/build-forwarders.sh)"; exit 2; }
tmp=$(mktemp -d "${TMPDIR:-/tmp}/vf-package-refusals.XXXXXX") || exit 2
trap 'rm -rf "$tmp"' EXIT
failed=0
# refused NAME WANTED: the copy in $tmp/NAME must be refused (status 2) with a line that has WANTED.
refused() {
    local name=$1 wanted=$2 log=$tmp/$1.log status
    DIST=$tmp/$name ALLOW_DIRTY=1 bash "$W/scripts/package.sh" > "$log" 2>&1
    status=$?
    if [ $status = 2 ] && grep -q -- "$wanted" "$log" && ! ls "$tmp/$name"/volta-forever-*.tar.* > /dev/null 2>&1; then
        echo "PASS  $name"
    else
        echo "FAIL  $name (status $status, wanted a refusal with: $wanted)"; sed 's/^/    /' "$log" | tail -5; failed=1
    fi
}
copy() {  # copy NAME: the built tree as it is (dates kept)
    mkdir -p "$tmp/$1" && cp -a "$W/dist/lib" "$W/dist/bin" "$W/dist/forward" "$tmp/$1/"
}
# bf16 NAME FILE...: forward/bf16tc of the copy NAME holds those files of the bf16 cuBLAS variants
# and not the others (empty stand-ins: package.sh refuses such a set by the names alone, before it
# reads a file; a file of the built tree is set aside).
bf16() {
    local name=$1 lib
    shift
    mkdir -p "$tmp/$name/forward/bf16tc"
    for lib in libcublas.so.13 libcublasLt.so.13 libvf_cublas_bf16.so; do
        if [ -e "$tmp/$name/forward/bf16tc/$lib" ]; then
            mv "$tmp/$name/forward/bf16tc/$lib" "$tmp/$name/forward/bf16tc/$lib.aside"
        fi
    done
    for lib do : > "$tmp/$name/forward/bf16tc/$lib"; done
}

copy missing-library
mv "$tmp/missing-library/forward/libcublasLt.so.13" "$tmp/missing-library/forward/libcublasLt.so.13.aside"
refused missing-library "lacks 1 of the .* forwarding libraries.*libcublasLt.so.13"

copy old-library
touch -d 2000-01-01 "$tmp/old-library/forward/libcufft.so.12"
refused old-library "libcufft.so.12 is older than"

copy partial-report
grep -v "^libcusolver.so.12 " "$W/dist/forward/REPORT.txt" > "$tmp/partial-report/forward/REPORT.txt"
touch -r "$W/dist/forward/REPORT.txt" "$tmp/partial-report/forward/REPORT.txt"
refused partial-report "lacks 1 of the .* forwarding libraries.*libcusolver.so.12"

copy bf16-variant-missing
bf16 bf16-variant-missing libcublas.so.13 libvf_cublas_bf16.so
refused bf16-variant-missing "bf16tc holds 2 of the 3 files of the bf16 cuBLAS variants: it lacks libcublasLt.so.13"

copy bf16-runtime-missing
bf16 bf16-runtime-missing libcublas.so.13 libcublasLt.so.13
refused bf16-runtime-missing "bf16tc holds 2 of the 3 files of the bf16 cuBLAS variants: it lacks libvf_cublas_bf16.so"

copy bf16-runtime-alone
bf16 bf16-runtime-alone libvf_cublas_bf16.so
refused bf16-runtime-alone "bf16tc holds 1 of the 3 files of the bf16 cuBLAS variants: it lacks libcublas.so.13 libcublasLt.so.13"

# The three files as built (the copy keeps forward/bf16tc of the built tree): package.sh checks the
# runtime with vf-fwdgen check-bf16, which empty stand-ins would not pass.
copy bf16-complete
absent=
for lib in libcublas.so.13 libcublasLt.so.13 libvf_cublas_bf16.so; do
    [ -s "$tmp/bf16-complete/forward/bf16tc/$lib" ] || absent="$absent $lib"
done
if [ -z "$absent" ]; then
    DIST=$tmp/bf16-complete ALLOW_DIRTY=1 bash "$W/scripts/package.sh" > "$tmp/bf16-complete.log" 2>&1
    status=$?
else
    echo "the built tree lacks forward/bf16tc/:$absent (scripts/build-forwarders.sh builds them)" > "$tmp/bf16-complete.log"
    status=absent
fi
got=$(tar -tzf "$tmp"/bf16-complete/volta-forever-*.tar.gz 2> /dev/null | sed -n 's|^[^/]*/forward/bf16tc/\(..*\)$|\1|p' | LC_ALL=C sort | tr '\n' ' ')
if [ $status = 0 ] && [ "$got" = "libcublas.so.13 libcublasLt.so.13 libvf_cublas_bf16.so " ]; then
    echo "PASS  bf16-complete"
else
    echo "FAIL  bf16-complete (status $status, forward/bf16tc of the archive: $got)"; sed 's/^/    /' "$tmp/bf16-complete.log" | tail -5; failed=1
fi

exit $failed
