# The directory where cargo builds a checkout, for the scripts that build or look for what cargo
# built (sourced; sets nothing by itself).
#
# vf_target_dir ROOT: $CARGO_TARGET_DIR when it is set; else a directory of this checkout's own,
# outside the tree: ${XDG_CACHE_HOME:-~/.cache}/volta-forever/target-HASH, HASH the first 12 hex
# digits of the SHA-256 of the checkout's physical path. Not one for every checkout: cargo tells
# whether what it built is fresh from the times of the files it read, at the paths it read them,
# and two checkouts at different paths build the same units (their hashes do not depend on the
# path). A second checkout whose files are older than the first one's outputs was given them: the
# forwarding generator embeds crates/vf-fwdgen/bf16tc/runtime.c (include_str!), and the second
# checkout's generator embedded the first one's.
vf_target_dir() {
    if [ -n "${CARGO_TARGET_DIR:-}" ]; then
        printf '%s\n' "$CARGO_TARGET_DIR"
        return
    fi
    local physical hash
    physical=$(cd "$1" && pwd -P) || return 1
    hash=$(printf '%s' "$physical" | sha256sum | cut -c1-12)
    printf '%s\n' "${XDG_CACHE_HOME:-$HOME/.cache}/volta-forever/target-$hash"
}
