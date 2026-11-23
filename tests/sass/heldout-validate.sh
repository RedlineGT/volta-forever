#!/bin/bash
# The prototype and the check on a corpus kept apart from the one the rules were found on:
# every sm_75 cubin under CORPUS (made by tests/sass/heldout-extract.sh) is translated by the
# reference translator (reference/batch.py) and its translation checked (reference/validate.py).
#
#   tests/sass/heldout-validate.sh CORPUS OUT [SUBDIRECTORY]
#
# Only the verdicts are kept, in OUT/checks (the translations and maps of the whole corpus are
# 19 GB): NAME.check (what validate.py said, and its exit status: 0 nothing found and everything
# checked, 1 problems, 2 nothing found and a check that could not be made), NAME.reasons (the
# functions by what became of them: `ok`, or why they were refused), NAME.lines (instructions in
# the map), NAME.failed (the prototype did not read the file). An hour for 3,108 cubins.
# The exit status is 1 when a translation has problems, when the validator could not make a check
# of one (its status 2) or ended without a verdict, or when no cubin was checked; a cubin that the
# prototype does not read at all is counted, and does not fail the run (both translators refuse
# it whole). (Until 2026-11-23 the status was 0 whatever the verdicts.)
#   JOBS   processes (default 40)
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
M=$root/reference
C=${1:?usage: heldout-validate.sh CORPUS OUT [SUBDIRECTORY]}
D=${2:?usage: heldout-validate.sh CORPUS OUT [SUBDIRECTORY]}
C=$C/${3:-}
JOBS=${JOBS:-40}
[ -e "$D" ] && { echo "$D exists: name a directory that does not" >&2; exit 2; }
mkdir -p "$D/checks" && cd "$D" || exit 2
set +m
find "$C" -name '*.sm_75.cubin' | xargs ls -S | awk -v jobs="$JOBS" '{print > ("list." (NR % jobs))}'
for f in list.*; do (
    t=$D/tmp.${f#list.}; mkdir -p "$t"
    for c in $(cat "$f"); do
        n=$(basename "$(dirname "$c")")__$(basename "$c")
        python3 -I "$M/batch.py" "$t" "$c" > "$t/log" 2>&1
        b=$t/$(basename "$c")
        if [ -e "$b" ]; then
            python3 -I "$M/validate.py" "$c" "$b" "$b.map" > "checks/$n.check" 2>&1; echo "exit $?" >> "checks/$n.check"
            cut -f1 "$b.report" 2>/dev/null | sort | uniq -c | sort -rn > "checks/$n.reasons"
            wc -l < "$b.map" > "checks/$n.lines"
        else
            { echo "no translation:"; tail -2 "$t/log"; } > "checks/$n.failed"
        fi
        find "$t" -maxdepth 1 -type f -delete
    done ) & done
wait
cat checks/*.check | grep "instructions of" | awk '{i+=$2; f+=$5; p+=$8} END {print i" instructions of "f" functions checked, "p" problems"}'
echo "cubins: $(find "$C" -name '*.sm_75.cubin' | wc -l); with a verdict: $(grep -l "instructions of" checks/*.check | wc -l); exit 0: $(grep -l "^exit 0" checks/*.check | wc -l), 1: $(grep -l "^exit 1" checks/*.check | wc -l), 2: $(grep -l "^exit 2" checks/*.check | wc -l); not read: $(ls checks/*.failed 2>/dev/null | wc -l); instructions in the maps: $(cat checks/*.lines | awk '{s+=$1} END {print s}')"
echo "functions: $(cat checks/*.reasons | awk '{s+=$1} END {print s}'), translated: $(cat checks/*.reasons | awk '$2 == "ok" {s+=$1} END {print s}')"
grep -h "^  [0-9]* (" checks/*.check | sed 's/^ *\([0-9]*\) /\1\t/' | awk -F'\t' '{n[$2]+=$1; f[$2]++} END {for (k in n) print "  " n[k], k, "(in", f[k], "cubins)"}' | sort -k2
grep -h "^  " checks/*.check | grep -v "^  [0-9]* (" | sed 's/^  [^:]*: //; s/: .*//' | sort | uniq -c | sort -rn | head -12
# (A verdict: the line that counts what was checked. Each NAME.check ends with the validator's status.)
failing=$({ grep -L "instructions of" checks/*.check; grep -l "^exit [12]$" checks/*.check; } 2>/dev/null | sort -u | wc -l)
[ -n "$(ls checks/*.check 2>/dev/null)" ] || { echo "no cubin was checked" >&2; exit 1; }
[ "$failing" -eq 0 ] || { echo "cubins whose translation has problems, a check not made or no verdict: $failing" >&2; exit 1; }
