#!/bin/bash
# The directory where cargo builds a checkout, for the scripts that build or look for what cargo
# built (sourced: it defines vf_target_dir and runs nothing); run, the list of those directories:
#   scripts/target-dir.sh --list
#
# vf_target_dir ROOT: $CARGO_TARGET_DIR when it is set; else a directory of this checkout's own,
# outside the tree: ${XDG_CACHE_HOME:-~/.cache}/volta-forever/target-HASH, HASH the first 12 hex
# digits of the SHA-256 of the checkout's physical path, in which it writes the file CHECKOUT with
# that path (so that a directory whose checkout is gone can be told, and removed: --list). Not one
# for every checkout: cargo tells whether what it built is fresh from the times of the files it
# read, at the paths it read them, and two checkouts at different paths build the same units (their
# hashes do not depend on the path). A second checkout whose files are older than the first one's
# outputs was given them: the forwarding generator embeds crates/vf-fwdgen/bf16tc/runtime.c
# (include_str!), and the second checkout's generator embedded the first one's. CARGO_TARGET_DIR
# shares one directory on purpose, with that staleness.
#
# --list: each target-HASH directory (and target, the one that every checkout shared before), its
# size, and its checkout: the path in its CHECKOUT, said ORPHAN when no directory is there any more.
vf_target_dir() {
    if [ -n "${CARGO_TARGET_DIR:-}" ]; then
        printf '%s\n' "$CARGO_TARGET_DIR"
        return
    fi
    local physical hash dir
    physical=$(cd "$1" && pwd -P) || return 1
    hash=$(printf '%s' "$physical" | sha256sum | cut -c1-12)
    dir=${XDG_CACHE_HOME:-$HOME/.cache}/volta-forever/target-$hash
    if [ "$(cat "$dir/CHECKOUT" 2>/dev/null)" != "$physical" ]; then
        { mkdir -p "$dir" && printf '%s\n' "$physical" >"$dir/CHECKOUT"; } 2>/dev/null || true
    fi
    printf '%s\n' "$dir"
}

if [ "${BASH_SOURCE:-}" = "$0" ]; then
    if [ "${1:-}" != --list ] || [ $# != 1 ]; then
        echo "usage: scripts/target-dir.sh --list   (sourced, it defines vf_target_dir)" >&2
        exit 2
    fi
    cache=${XDG_CACHE_HOME:-$HOME/.cache}/volta-forever
    listed=0
    for dir in "$cache"/target "$cache"/target-*; do
        [ -d "$dir" ] || continue
        listed=1
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        if [ "$dir" = "$cache/target" ]; then
            what="the directory that every checkout shared before target-HASH (used only when CARGO_TARGET_DIR names it)"
        elif [ ! -f "$dir/CHECKOUT" ]; then
            what="no CHECKOUT (made before it was written: the next build of its checkout writes it)"
        else
            checkout=$(cat "$dir/CHECKOUT")
            if [ -d "$checkout" ]; then
                what="of $checkout"
            else
                what="ORPHAN: its checkout $checkout is not there any more (the directory can be removed)"
            fi
        fi
        printf '%s\t%s\t%s\n' "${size:-?}" "$dir" "$what"
    done
    [ "$listed" = 1 ] || echo "no target directory in $cache"
    exit 0
fi
