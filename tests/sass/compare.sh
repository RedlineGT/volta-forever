#!/bin/bash
# The Rust translator (crates/vf-sass) against the Python reference (reference/translate.py): for every
# cubin, the translated bytes, the map of instructions, what the validator reads beside the map
# (MAP.lives) and the refusals with their reasons must be the same.
#
#   tests/sass/compare.sh [CUBIN | DIR]...     (cubins of different names: one directory a call)
#
# Without arguments: the sm_75 cubins of the test kernels (tests/kernels/translate), and those of
# PyTorch when they are where `TORCH` says (extracted with `vf extract libtorch_cuda.so DIR 75`).
# The prototype's outputs are made here (48 processes, a minute for PyTorch's 475 cubins), or taken
# from the directory `REFERENCE` names (outputs of batch.py) when it has them.
#   VF_TOOL  the tool. By default the one that `cargo build --release -p vf-cli` built, where
#              cargo put it: release/vf-translate of $CARGO_TARGET_DIR when that is set, or else of this
#              tree's target/ when it is there, or else of the directory scripts/build.sh builds
#              in (~/.cache/volta-forever/target)
#   JOBS       processes (default 48)
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$root/tests/site.env" ] && . "$root/tests/site.env"
: "${VF_TEST_ROOT:=${XDG_CACHE_HOME:-$HOME/.cache}/volta-forever/tests}"
md=$root/reference
if [ -z "${VF_TOOL:-}" ]; then
    if [ -n "${CARGO_TARGET_DIR:-}" ]; then VF_TOOL=$CARGO_TARGET_DIR/release/vf-translate
    elif [ -x "$root/target/release/vf-translate" ]; then VF_TOOL=$root/target/release/vf-translate
    else VF_TOOL=${XDG_CACHE_HOME:-$HOME/.cache}/volta-forever/target/release/vf-translate
    fi
fi
TORCH=${TORCH:-$VF_TEST_ROOT/torch-cubins}
JOBS=${JOBS:-48}
[ -x "$VF_TOOL" ] || { echo "no tool at $VF_TOOL: cargo build --release -p vf-cli, or name it with VF_TOOL=" >&2; exit 2; }

inputs=()
if [ $# -eq 0 ]; then
    set -- "$root/tests/kernels/translate"
    [ -d "$TORCH" ] && set -- "$@" "$TORCH"
fi
for arg in "$@"; do
    if [ -d "$arg" ]; then
        for f in "$arg"/*.cubin; do
            case $f in *.sm70.cubin) ;; *) [ -f "$f" ] && inputs+=("$f") ;; esac
        done
    else
        inputs+=("$arg")
    fi
done
[ ${#inputs[@]} -gt 0 ] || { echo "no cubin to translate" >&2; exit 2; }
# The outputs go by the cubin's name alone: two of the same name (the held-out corpus has a 0000.sm_75.cubin
# in each library's directory) would be compared with each other's. One directory a call.
twice=$(printf '%s\n' "${inputs[@]}" | sed 's|.*/||' | sort | uniq -d | head -1)
[ -z "$twice" ] || { echo "two cubins are named $twice: one directory a call" >&2; exit 2; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir "$work/py" "$work/rs"

# The prototype: batch.py writes NAME, NAME.map, NAME.map.lives and NAME.report (reasons without
# their offsets).
todo=()
for f in "${inputs[@]}"; do
    name=$(basename "$f")
    if [ -n "${REFERENCE:-}" ] && [ -f "$REFERENCE/$name.report" ]; then
        for x in "" .map .map.lives .report; do [ -f "$REFERENCE/$name$x" ] && ln -s "$REFERENCE/$name$x" "$work/py/$name$x"; done
    else
        todo+=("$f")
    fi
done
if [ ${#todo[@]} -gt 0 ]; then
    printf '%s\n' "${todo[@]}" | awk -v jobs="$JOBS" -v dir="$work" '{print > (dir "/list." (NR % jobs))}'
    for list in "$work"/list.*; do
        python3 -I "$md/batch.py" "$work/py" $(cat "$list") > "$list.log" 2>&1 &
    done
    wait
fi

# The Rust translator.
printf '%s\n' "${inputs[@]}" | xargs -P "$JOBS" -I{} sh -c '
    name=$(basename "$1"); out=$2/rs
    "$3" translate "$1" "$out/$name" "$out/$name.map" "$out/$name.raw" > "$out/$name.out" 2> "$out/$name.err"
    echo $? > "$out/$name.rc"
    # Reasons without their offsets, as batch.py writes them.
    [ -f "$out/$name.raw" ] && awk -F"\t" "BEGIN { OFS = FS } { gsub(/0x[0-9a-f]+/, \"N\", \$1); print }" "$out/$name.raw" > "$out/$name.report"
' sh {} "$work" "$VF_TOOL"

total=0; same=0; functions=0; refused=0
for f in "${inputs[@]}"; do
    name=$(basename "$f"); total=$((total + 1)); what=""
    if grep -qs '^CRASH' "$work/py/$name.report"; then
        # The prototype stopped on the file: nothing to compare with. The translator must still end properly.
        [ "$(cat "$work/rs/$name.rc" 2>/dev/null)" -le 1 ] 2>/dev/null || what="$what the translator ended badly where the prototype crashed;"
    else
        [ "$(cat "$work/rs/$name.rc" 2>/dev/null)" = 0 ] || what="$what exit $(cat "$work/rs/$name.rc" 2>/dev/null);"
        cmp -s "$work/rs/$name" "$work/py/$name" || what="$what bytes;"
        cmp -s "$work/rs/$name.map" "$work/py/$name.map" || what="$what map;"
        cmp -s "$work/rs/$name.map.lives" "$work/py/$name.map.lives" || what="$what lives;"
        cmp -s "$work/rs/$name.report" "$work/py/$name.report" || what="$what refusals;"
    fi
    if [ -z "$what" ]; then
        same=$((same + 1))
    else
        echo "$name:$what $(head -c 200 "$work/rs/$name.err" 2>/dev/null)"
    fi
    if [ -f "$work/py/$name.report" ]; then
        functions=$((functions + $(wc -l < "$work/py/$name.report")))
        refused=$((refused + $(grep -vc '^ok	' "$work/py/$name.report")))
    fi
done
echo "$same of $total cubins the same as the reference's ($functions functions, $refused refused)"
[ "$same" -eq "$total" ]
