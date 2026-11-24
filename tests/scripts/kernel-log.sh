#!/bin/bash
# tests/gpu/kernel-log.sh, the source of the kernel's log of the GPU scripts, without a GPU and
# without reading this machine's log: stand-ins for dmesg, journalctl and sudo, first in PATH, give
# each source or refuse it, as they do for a user:
#   tests/scripts/kernel-log.sh
# - dmesg readable: it is the source;
# - dmesg refused (kernel.dmesg_restrict=1), the journal readable (a member of adm or
#   systemd-journal): journalctl -k;
# - the journal giving no line either (status 0, as to a user outside those groups), sudo without
#   a password: sudo -n dmesg, and sudo is never run without -n;
# - none: no source, kernel_log fails, and the Xid lines are counted as 0.
# In each case the lines with "Xid" are those of the source, and the other commands give none.
# Prints PASS or FAIL for each case; exit status 1 on a FAIL.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
tmp=$(mktemp -d "${TMPDIR:-/tmp}/vf-kernel-log.XXXXXX") || exit 2
trap 'rm -rf "$tmp"' EXIT
failed=0

# stub DIRECTORY NAME STATUS [LINES]: a command NAME that prints LINES and ends with STATUS, and
# writes its arguments to DIRECTORY/NAME.called.
stub() {
    local dir=$1 name=$2 status=$3 lines=${4:-}
    mkdir -p "$dir"
    cat > "$dir/$name" <<EOF
#!/bin/bash
echo "\$*" >> "$dir/$name.called"
[ -z "$lines" ] || printf '%s\n' $lines
[ $status = 0 ] || echo "$name: refused (a stand-in)" >&2
exit $status
EOF
    chmod +x "$dir/$name"
}
# The kernel's log of each source: two lines with Xid in dmesg's, three in the journal's, four in
# what sudo gives (as one word each, for printf).
from_dmesg="kernel:a NVRM:Xid:31 NVRM:Xid:43"
from_journal="kernel:b NVRM:Xid:31 NVRM:Xid:43 NVRM:Xid:79"
from_sudo="kernel:c NVRM:Xid:31 NVRM:Xid:43 NVRM:Xid:79 NVRM:Xid:13"

# check NAME SOURCE XIDS DMESG JOURNAL SUDO: with the stand-ins given as STATUS:LINES (LINES the
# name of one of the variables above, or empty), the source chosen is SOURCE and the log has XIDS
# lines with Xid.
check() {
    local name=$1 source=$2 xids=$3 dir=$tmp/$1 got count status
    shift 3
    local i=0 command
    for command in dmesg journalctl sudo; do
        i=$((i + 1))
        local spec=${!i}
        local lines=${spec#*:}
        stub "$dir" "$command" "${spec%%:*}" "${lines:+${!lines}}"
    done
    got=$(PATH="$dir:$PATH" bash -c '. "$1"; printf "%s|" "$kernel_log_source"; kernel_log | grep -c Xid; echo "|${PIPESTATUS[0]}"' _ \
        "$W/tests/gpu/kernel-log.sh")
    count=$(echo "$got" | tr -d '\n' | cut -d'|' -f2)
    status=$(echo "$got" | tr -d '\n' | cut -d'|' -f3)
    local ok=1 why=()
    [ "${got%%|*}" = "$source" ] || { ok=0; why+=("source: '${got%%|*}', expected '$source'"); }
    [ "$count" = "$xids" ] || { ok=0; why+=("lines with Xid: $count, expected $xids"); }
    if [ -z "$source" ] && [ "$status" = 0 ]; then ok=0; why+=("kernel_log succeeded without a source"); fi
    if [ -f "$dir/sudo.called" ] && grep -qv '^-n ' "$dir/sudo.called"; then
        ok=0; why+=("sudo was run without -n: $(tr '\n' ';' < "$dir/sudo.called")")
    fi
    if [ $ok = 1 ]; then
        echo "PASS  $name"
    else
        echo "FAIL  $name"; printf '    %s\n' "${why[@]}"; failed=1
    fi
}

check dmesg-readable dmesg 2 0:from_dmesg 0:from_journal 0:from_sudo
check journal-readable "journalctl -k" 3 1: 0:from_journal 0:from_sudo
check sudo-without-password "sudo -n dmesg" 4 1: 0: 0:from_sudo
check none-readable "" 0 1: 0: 1:
# A journal that refuses outright (status 1) is passed over as one that gives nothing.
check journal-refused "sudo -n dmesg" 4 1: 1: 0:from_sudo
exit $failed
