#!/bin/bash
# Run a command in a home of its own, and fail when the command leaves anything there:
#   tests/scripts/untouched-home.sh COMMAND [ARG...]
#
# COMMAND runs with HOME at an empty directory made for it (under TMPDIR, removed at the end),
# XDG_CACHE_HOME, XDG_CONFIG_HOME, XDG_DATA_HOME and XDG_STATE_HOME under it, and VF_PROXY_CACHE
# unset, so that the layer's cache is where it is by default; CARGO_HOME and RUSTUP_HOME stay where
# they were, for cargo to find its toolchain and its crates. The CI and tests/release-check.sh run
# the unit tests so: what they leave there, they would leave in the home of whoever runs them (until
# 2026-11-24, one test kept its translation of machine code in ~/.cache/volta-forever/proxy at each
# run). The exit status is COMMAND's when it is not 0, else 1 when anything was created in that
# home (it is listed), else 0.
set -uo pipefail
if [ $# = 0 ]; then
    echo "usage: $0 COMMAND [ARG...]" >&2
    exit 2
fi
home=$(mktemp -d "${TMPDIR:-/tmp}/vf-home.XXXXXX") || exit 2
trap 'rm -rf "$home"' EXIT
env -u VF_PROXY_CACHE CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}" RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}" \
    HOME="$home" XDG_CACHE_HOME="$home/.cache" XDG_CONFIG_HOME="$home/.config" \
    XDG_DATA_HOME="$home/.local/share" XDG_STATE_HOME="$home/.local/state" "$@"
status=$?
left=$(cd "$home" && find . -mindepth 1 -printf '%y %10s  %P\n' | sort -k3)
if [ -n "$left" ]; then
    echo "untouched-home.sh: the command left this in its home (type, size, path):"
    printf '%s\n' "$left" | sed 's/^/  /'
    [ $status != 0 ] || status=1
else
    echo "untouched-home.sh: the command left nothing in its home"
fi
exit $status
