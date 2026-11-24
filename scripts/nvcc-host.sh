#!/bin/bash
# What a CUDA compiler needs on this machine to compile CUDA C++, found by compiling a probe with
# it (never from version numbers): nothing on most systems. On a system newer than the toolkit, a
# host compiler that the toolkit accepts in place of the default one, or the C library's
# declarations that the toolkit's headers contradict hidden (nvcc-host/bits/mathcalls.h says
# which and why), or both: CUDA 12.9 with GCC 15 and glibc 2.43 needs both. Every script of the
# repository that runs nvcc goes through here.
#   . scripts/nvcc-host.sh                    in a bash script, then:
#     nvcc_host NVCC ARGUMENTS...             runs NVCC with what it needs, then ARGUMENTS
#     nvcc_host_args NVCC                     sets the array NVCC_HOST_ARGS to what it needs
#   scripts/nvcc-host.sh NVCC ARGUMENTS...    runs NVCC with what it needs, then ARGUMENTS
#   scripts/nvcc-host.sh --args NVCC          prints what it needs, one argument per line
# The probe is a CUDA source that calls rsqrt, sinpi and cospi in device and host code, compiled
# to an object (nvcc -c). Tried in this order until it compiles: NVCC as it is (then nothing is
# added, and the command is the one it would be without this script); with -isystem nvcc-host;
# then, for each g++-N in PATH from the newest down, -ccbin g++-N without and with -isystem
# nvcc-host. Other host compilers are not tried when one is chosen already (NVCC_CCBIN, or -ccbin
# or --compiler-bindir in NVCC_PREPEND_FLAGS or NVCC_APPEND_FLAGS). What is added is said on
# standard error, once for each NVCC in a shell, which keeps the answer. When nothing compiles the
# probe, that is said with the probe's first error, and nothing is added: NVCC then fails as it
# would without this script.

_nvcc_host_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/nvcc-host
declare -gA _nvcc_host_known=()

# _nvcc_host_probe NVCC ARGUMENTS... : 0 when NVCC (an absolute path) compiles the probe with
# ARGUMENTS; the compiler's output in $_nvcc_host_out.
_nvcc_host_probe() {
    local work status
    work=$(mktemp -d) || { _nvcc_host_out="no temporary directory for the probe (mktemp -d)"; return 2; }
    cat > "$work/probe.cu" <<'EOF'
#include <cmath>
__global__ void vf_probe(float *f, double *d)
{
    f[0] = rsqrtf(f[0]) + sinpif(f[1]) + cospif(f[2]);
    d[0] = rsqrt(d[0]) + sinpi(d[1]) + cospi(d[2]);
}
double vf_probe_host(double x) { return rsqrt(x) + sinpi(x) + cospi(x); }
EOF
    _nvcc_host_out=$(cd "$work" && "$@" -c -o probe.o probe.cu 2>&1)
    status=$?
    rm -rf "$work"
    return $status
}

# The g++-N of PATH (absolute directories), one path a line, from the highest N down; for each N
# the first in PATH.
_nvcc_host_compilers() {
    local d f
    local IFS=:
    for d in $PATH; do
        case $d in /*) ;; *) continue ;; esac
        for f in "$d"/g++-[0-9]*; do
            [[ -x $f && ${f##*/} =~ ^g\+\+-([0-9]+)$ ]] && echo "${BASH_REMATCH[1]} $f"
        done
    done | sort -s -u -n -k1,1 | sort -s -nr -k1,1 | cut -d' ' -f2-
}

nvcc_host_args() {
    local nvcc=$1 first chosen=0 compiler version default_version tried= why
    local -a candidates=("") try=()
    case $nvcc in
        /*) ;;
        */*) nvcc=$PWD/$nvcc ;;
        *) nvcc=$(command -v "$nvcc") || nvcc=$1 ;;
    esac
    NVCC_HOST_ARGS=()
    if [ -n "${_nvcc_host_known[$nvcc]+set}" ]; then
        [ -z "${_nvcc_host_known[$nvcc]}" ] || mapfile -t NVCC_HOST_ARGS <<< "${_nvcc_host_known[$nvcc]}"
        return 0
    fi
    _nvcc_host_known[$nvcc]=
    _nvcc_host_probe "$nvcc" && return 0
    first=$(grep -m1 -i 'error' <<< "$_nvcc_host_out" || head -n1 <<< "$_nvcc_host_out")
    # The host compilers after the default one: g++-N, without those of the default's version.
    case " ${NVCC_PREPEND_FLAGS:-} ${NVCC_APPEND_FLAGS:-} " in
        *" -ccbin"* | *" --compiler-bindir"*) chosen=1 ;;
    esac
    [ -z "${NVCC_CCBIN:-}" ] || chosen=1
    if [ "$chosen" = 0 ]; then
        default_version=$(gcc -dumpfullversion 2>/dev/null) || default_version=
        while IFS= read -r compiler; do
            version=$("$compiler" -dumpfullversion 2>/dev/null) || continue
            [ "$version" = "$default_version" ] || candidates+=("$compiler")
        done < <(_nvcc_host_compilers)
    fi
    for compiler in "${candidates[@]}"; do
        try=()
        [ -z "$compiler" ] || try=(-ccbin "$compiler")
        if [ -n "$compiler" ] && _nvcc_host_probe "$nvcc" "${try[@]}"; then
            :
        elif _nvcc_host_probe "$nvcc" "${try[@]}" -isystem "$_nvcc_host_dir"; then
            try+=(-isystem "$_nvcc_host_dir")
        else
            tried="$tried, ${compiler:-the default host compiler}"
            continue
        fi
        NVCC_HOST_ARGS=("${try[@]}")
        _nvcc_host_known[$nvcc]=$(printf '%s\n' "${try[@]}")
        why=
        [ -z "$compiler" ] || why="the probe does not compile with the default host compiler"
        case " ${try[*]} " in
            *" -isystem "*) why="${why:+$why; }the C library declares rsqrt, sinpi or cospi otherwise than the toolkit's headers: scripts/nvcc-host/bits/mathcalls.h hides those declarations" ;;
        esac
        echo "nvcc-host: $nvcc: adding ${try[*]} ($why)" >&2
        return 0
    done
    echo "nvcc-host: $nvcc compiles no CUDA C++ here: not with ${tried#, }, with the C library's declarations of rsqrt, sinpi and cospi hidden or not; nothing is added. Its first error: $first. If that is about the host compiler, install a g++ that this CUDA release accepts (its include/crt/host_config.h names the newest GCC) under the name g++-N in PATH." >&2
    return 0
}

nvcc_host() {
    nvcc_host_args "$1"
    "$1" "${NVCC_HOST_ARGS[@]}" "${@:2}"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    case ${1:-} in
        --args)
            [ $# = 2 ] || { echo "usage: $0 --args NVCC" >&2; exit 2; }
            nvcc_host_args "$2"
            [ ${#NVCC_HOST_ARGS[@]} = 0 ] || printf '%s\n' "${NVCC_HOST_ARGS[@]}" ;;
        '' | -h | --help) sed -n '2,/^$/{/^$/d;s/^# \{0,1\}//;p}' "$0"; [ $# = 1 ] ;;
        *)
            nvcc_host_args "$1"
            exec "$1" "${NVCC_HOST_ARGS[@]}" "${@:2}" ;;
    esac
fi
