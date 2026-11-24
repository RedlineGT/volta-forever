# The kernel's log, for the GPU scripts that count its lines with "Xid" before and after cases that
# could stop the GPU (tests/gpu/run.sh). Sourced: it runs nothing but the choice of a source, and
# defines
#   kernel_log_source  the first of these that can be read here, empty when none can:
#                        dmesg          the kernel's own buffer, which Ubuntu keeps from users
#                                       (kernel.dmesg_restrict=1);
#                        journalctl -k  the kernel's lines in the journal (the current boot's),
#                                       readable by the groups adm and systemd-journal: read when it
#                                       gives at least one line (it gives none, with status 0, to a
#                                       user outside those groups);
#                        sudo -n dmesg  where sudo needs no password (-n: it never asks for one);
#   kernel_log         prints the log from that source (status 1 when there is none). journald
#                      forwards what the kernel writes a moment later: the journal is read after a
#                      second.
# tests/scripts/kernel-log.sh tests it, with stand-ins for the three commands.
kernel_log_source=
if dmesg >/dev/null 2>&1; then
    kernel_log_source=dmesg
elif [ -n "$(journalctl -k -q --no-pager -n 1 2>/dev/null)" ]; then
    kernel_log_source="journalctl -k"
elif sudo -n dmesg >/dev/null 2>&1; then
    kernel_log_source="sudo -n dmesg"
fi

kernel_log() {
    case $kernel_log_source in
        dmesg) dmesg 2>/dev/null ;;
        "journalctl -k") sleep 1 && journalctl -k -q --no-pager 2>/dev/null ;;
        "sudo -n dmesg") sudo -n dmesg 2>/dev/null ;;
        *) return 1 ;;
    esac
}
