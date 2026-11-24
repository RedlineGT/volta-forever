#!/bin/bash
# Where the build scripts have cargo build a checkout, without a GPU:
#   tests/scripts/target-dir.sh [--build]
# - two checkouts (copies of scripts/ in two directories) are built in two directories of their
#   own, and each script of a checkout builds it in the same one: scripts/build.sh and
#   scripts/build-forwarders.sh, run with a stand-in for cargo that says where it was told to
#   build; CARGO_TARGET_DIR, when it is set, is where they build; each directory of a checkout has
#   the file CHECKOUT with the checkout's path, and scripts/target-dir.sh --list says each directory
#   with its checkout, and ORPHAN when that checkout is gone; scripts/build.sh has rustc remap the
#   paths of the crates (CARGO_HOME, else ~/.cargo), of the build directory and of the checkout,
#   so that no path of the build host is in the binaries and a build under another HOME gives the
#   same bytes;
# - with --build, two copies of this tree, the second's crates/vf-fwdgen/bf16tc/runtime.c changed
#   and all its files older than what the first's build made, each built by
#   scripts/build-forwarders.sh (with cargo, as a user's checkouts are): each checkout's generator
#   embeds its own runtime.c. (With one directory for every checkout, the second was given the
#   first one's generator: cargo found its outputs fresh.)
# Prints PASS or FAIL for each check; exit status 1 on a FAIL.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
tmp=$(mktemp -d "${TMPDIR:-/tmp}/vf-target-dir.XXXXXX") || exit 2
trap 'rm -rf "$tmp"' EXIT
failed=0
check() {  # check NAME CONDITION...
    local name=$1; shift
    if "$@"; then echo "PASS  $name"; else echo "FAIL  $name"; failed=1; fi
}

# The stand-in for cargo: writes where it was told to build and the flags it was given for rustc, and
# stops the script.
printf '#!/bin/sh\necho "$CARGO_TARGET_DIR" > "$SEEN"\necho " $RUSTFLAGS " > "$SEEN.flags"\nexit 3\n' > "$tmp/cargo"
chmod +x "$tmp/cargo"
for checkout in a b; do
    mkdir -p "$tmp/$checkout/scripts"
    cp "$W/scripts/build.sh" "$W/scripts/build-forwarders.sh" "$W/scripts/target-dir.sh" "$tmp/$checkout/scripts/"
done
seen() {  # seen CHECKOUT SCRIPT [VAR=VALUE...]: where SCRIPT of CHECKOUT has cargo build
    local checkout=$1 script=$2; shift 2
    rm -f "$tmp/seen" "$tmp/seen.flags"
    env -u CARGO_TARGET_DIR XDG_CACHE_HOME="$tmp/cache" CARGO="$tmp/cargo" SEEN="$tmp/seen" "$@" \
        "$tmp/$checkout/scripts/$script" >/dev/null 2>&1
    cat "$tmp/seen" 2>/dev/null
}
a=$(seen a build.sh)
a_forwarders=$(seen a build-forwarders.sh)
b=$(seen b build.sh)
given=$(seen b build.sh CARGO_TARGET_DIR="$tmp/given")
echo "  checkout a: $a; checkout b: $b"
check "one directory for each checkout" test -n "$a" -a -n "$b" -a "$a" != "$b"
check "the same for each script of a checkout" test -n "$a" -a "$a" = "$a_forwarders"
case $a in "$tmp/cache/volta-forever/target-"*) inside=1 ;; *) inside=0 ;; esac
check "under XDG_CACHE_HOME, outside the tree" test "$inside" = 1
check "CARGO_TARGET_DIR when it is set" test "$given" = "$tmp/given"
check "each directory names its checkout" test "$(cat "$a/CHECKOUT" 2>/dev/null)" = "$(cd "$tmp/a" && pwd -P)" \
    -a "$(cat "$b/CHECKOUT" 2>/dev/null)" = "$(cd "$tmp/b" && pwd -P)"
check "none written where CARGO_TARGET_DIR names" test ! -e "$tmp/given/CHECKOUT"
remaps() {  # remaps CARGO_HOME: whether build.sh, run with HOME=$tmp/home, remaps the crates of
    # CARGO_HOME (an empty one: of ~/.cargo), the build directory and the checkout
    local crates=${1:-$tmp/home/.cargo} dir flags
    dir=$(seen a build.sh HOME="$tmp/home" CARGO_HOME="$1")
    flags=$(cat "$tmp/seen.flags" 2>/dev/null)
    echo "  CARGO_HOME=${1:-(empty)}: RUSTFLAGS=$flags"
    case $flags in *" --remap-path-prefix=$crates=cargo "*) ;; *) return 1 ;; esac
    case $flags in *" --remap-path-prefix=$dir=target "*) ;; *) return 1 ;; esac
    case $flags in *" --remap-path-prefix=$tmp/a=volta-forever "*) ;; *) return 1 ;; esac
}
check "build.sh remaps the crates of CARGO_HOME, not of HOME, the build directory and the checkout" \
    remaps "$tmp/crates"
check "build.sh remaps the crates of ~/.cargo when CARGO_HOME is not set" remaps ""
rm -rf "$tmp/b"
listed=$(XDG_CACHE_HOME="$tmp/cache" "$tmp/a/scripts/target-dir.sh" --list 2>&1)
echo "$listed" | sed 's/^/  /'
of_a=$(printf '%s\n' "$listed" | grep -c "$a	of $(cd "$tmp/a" && pwd -P)$")
of_b=$(printf '%s\n' "$listed" | grep -c "$b	ORPHAN: its checkout .*/b is not there any more")
check "--list: each directory with its checkout, ORPHAN when it is gone" test "$of_a" = 1 -a "$of_b" = 1

if [ "${1:-}" = --build ]; then
    cargo=${CARGO:-$(command -v cargo || echo "$HOME/.cargo/bin/cargo")}
    for checkout in a b; do
        mkdir -p "$tmp/tree-$checkout"
        (cd "$W" && git ls-files -z | tar --null -T - -c) | tar -x -C "$tmp/tree-$checkout" || exit 2
        echo "/* checkout-$checkout */" >> "$tmp/tree-$checkout/crates/vf-fwdgen/bf16tc/runtime.c"
    done
    # (The second checkout's files older than the first's outputs.)
    find "$tmp/tree-b" -exec touch -h -d 2020-01-01 {} +
    embedded() {  # embedded CHECKOUT: the marker in the generator its build-forwarders.sh built
        local dir
        env -u CARGO_TARGET_DIR -u RUSTFLAGS XDG_CACHE_HOME="$tmp/cache" CARGO="$cargo" \
            "$tmp/tree-$1/scripts/build-forwarders.sh" "$tmp/forward-$1" >/dev/null 2>&1
        dir=$(unset CARGO_TARGET_DIR; . "$tmp/tree-$1/scripts/target-dir.sh" && XDG_CACHE_HOME="$tmp/cache" vf_target_dir "$tmp/tree-$1")
        grep -ao "checkout-[ab]" "$dir/release/vf-fwdgen" 2>/dev/null | sort -u | tr '\n' ' '
    }
    first=$(embedded a)
    second=$(embedded b)
    echo "  the generator of checkout a embeds: $first; of checkout b: $second"
    check "each checkout's generator embeds its own runtime.c" test "$first" = "checkout-a " -a "$second" = "checkout-b "
fi
exit $failed
