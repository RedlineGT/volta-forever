#!/bin/bash
# scripts/nvcc-host.sh with the CUDA compilers of toolkits/ (12.9.2, and 13.5.0 when it is there),
# without a GPU:
#   tests/scripts/nvcc-host.sh
# - this machine as it is: when nvcc compiles the probe as it is, the helper adds nothing and says
#   nothing (the scripts run the commands they ran before it); when it does not (a system newer
#   than the toolkit, such as Ubuntu 26.04 with glibc 2.43 and GCC 15, for CUDA 12.9.2), the
#   helper finds what makes it compile, and a program that calls rsqrt, sinpi and cospi in host
#   and device code is compiled with that, runs, and prints the right values. The second half
#   runs only on such a system;
# - a C library that declares rsqrt, sinpi and cospi as glibc 2.43 does, on any machine: a
#   directory whose bits/mathcalls.h adds those declarations, given to nvcc after the helper's
#   (NVCC_APPEND_FLAGS). Without nvcc-host/, the program does not compile ("exception
#   specification is incompatible", what the toolkit's headers and glibc 2.43 give); the helper
#   adds -isystem nvcc-host, and the program then compiles, runs and prints the right values;
# - a default host compiler that CUDA 12.9.2 refuses: a gcc first in PATH that says it is GCC 99.
#   The helper chooses a g++-N of PATH (-ccbin), and the program compiles and runs with it.
#   Skipped when PATH has no g++-N.
# Prints PASS, FAIL or SKIP for each case; exit status 1 on a FAIL, 2 without toolkits/cuda-12.9.2.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
helper=$W/scripts/nvcc-host.sh
ours=$W/scripts/nvcc-host
nvcc12=$W/toolkits/cuda-12.9.2/bin/nvcc
nvcc13=$W/toolkits/cuda-13.5.0/bin/nvcc
[ -x "$nvcc12" ] || { echo "missing: toolkits/cuda-12.9.2 (scripts/fetch-all.sh redist)"; exit 2; }
tmp=$(mktemp -d "${TMPDIR:-/tmp}/vf-nvcc-host.XXXXXX") || exit 2
trap 'rm -rf "$tmp"' EXIT
failed=0
pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; shift; [ $# = 0 ] || printf '    %s\n' "$@"; failed=1; }

# The program: host results of CUDA's rsqrt, sinpi and cospi, double and float, and a kernel that
# calls them.
cat > "$tmp/math.cu" <<'EOF'
#include <cmath>
#include <cstdio>
__global__ void kernel(float *f, double *d)
{
    f[0] = rsqrtf(f[0]) + sinpif(f[1]) + cospif(f[2]);
    d[0] = rsqrt(d[0]) + sinpi(d[1]) + cospi(d[2]);
}
int main()
{
    std::printf("%g %g %g %g %g %g\n", rsqrt(4.0), sinpi(0.5), cospi(1.0), rsqrtf(4.0f), sinpif(0.5f), cospif(1.0f));
    return 0;
}
EOF
want="0.5 1 -1 0.5 1 -1"
# build NAME NVCC ARGUMENTS...: compile and run the program; 0 when it prints $want. Its output in
# $tmp/NAME.log.
build() {
    local name=$1
    shift
    (cd "$tmp" && "$@" -Wno-deprecated-gpu-targets -o "$name" math.cu) > "$tmp/$name.log" 2>&1 &&
        [ "$("$tmp/$name" 2>&1)" = "$want" ]
}
# args NAME NVCC: what the helper adds for NVCC into the array added, and what it says into
# $tmp/NAME.said.
args() {
    mapfile -t added < <(bash "$helper" --args "$2" 2> "$tmp/$1.said")
}

# ---- this machine as it is
plain=1
for nvcc in "$nvcc12" "$nvcc13"; do
    [ -x "$nvcc" ] || continue
    v=${nvcc%/bin/nvcc}
    v=${v##*/cuda-}
    args "here-$v" "$nvcc"
    if [ ${#added[@]} = 0 ]; then
        if [ -s "$tmp/here-$v.said" ]; then
            fail "here-$v (nothing added, but it said something)" "$(cat "$tmp/here-$v.said")"
        elif build "here-$v" "$nvcc"; then
            pass "here-$v (nvcc $v compiles as it is: nothing added, nothing said)"
        else
            fail "here-$v (nothing added, and the program does not build or print $want)" "$(tail -3 "$tmp/here-$v.log")"
        fi
    else
        [ "$v" != 12.9.2 ] || plain=0
        if grep -q "^nvcc-host: .*: adding ${added[*]} (" "$tmp/here-$v.said" && build "here-$v" "$nvcc" "${added[@]}"; then
            pass "here-$v (nvcc $v compiles here with ${added[*]})"
        else
            fail "here-$v (added ${added[*]})" "$(cat "$tmp/here-$v.said")" "$(tail -3 "$tmp/here-$v.log")"
        fi
    fi
done

# ---- a C library that declares the three functions as glibc 2.43 does
mkdir -p "$tmp/c-library/bits"
cat > "$tmp/c-library/bits/mathcalls.h" <<'EOF'
/* bits/mathcalls.h of the C library, then glibc 2.43's declarations of sinpi, cospi and rsqrt
   (ISO C23, declared __THROW) for double and float: what <math.h> declares there whatever the
   C library of the machine. */
#include_next <bits/mathcalls.h>
#define __vf_test_precision_ 1
#define __vf_test_precision_f 1
#if __MATH_PRECNAME (__vf_test_precision_, )
# ifndef __DECL_SIMD_sinpi
#  define __DECL_SIMD_sinpi
#  define __DECL_SIMD_sinpif
#  define __DECL_SIMD_cospi
#  define __DECL_SIMD_cospif
#  define __DECL_SIMD_rsqrt
#  define __DECL_SIMD_rsqrtf
# endif
__MATHCALL_VEC (sinpi,, (_Mdouble_ __x));
__MATHCALL_VEC (cospi,, (_Mdouble_ __x));
__MATHCALL_VEC (rsqrt,, (_Mdouble_ __x));
#endif
#undef __vf_test_precision_
#undef __vf_test_precision_f
EOF
export NVCC_APPEND_FLAGS="-isystem $tmp/c-library"
args libc "$nvcc12"
unset NVCC_APPEND_FLAGS
if [ ${#added[@]} -lt 2 ] || [ "${added[-2]}" != -isystem ] || [ "${added[-1]}" != "$ours" ]; then
    fail "libc (added ${added[*]}, not -isystem $ours last)" "$(cat "$tmp/libc.said")"
else
    # The control: what the helper added, without nvcc-host.
    control=("${added[@]:0:${#added[@]}-2}")
    if NVCC_APPEND_FLAGS="-isystem $tmp/c-library" build libc-control "$nvcc12" "${control[@]}"; then
        fail "libc (the control compiles: the declarations do not disagree)" "$(tail -3 "$tmp/libc-control.log")"
    elif ! grep -q 'exception specification is incompatible with that of previous function "rsqrt"' "$tmp/libc-control.log"; then
        fail "libc (the control fails for another reason)" "$(grep -m3 error "$tmp/libc-control.log")"
    elif ! NVCC_APPEND_FLAGS="-isystem $tmp/c-library" build libc "$nvcc12" "${added[@]}"; then
        fail "libc (added ${added[*]}, and the program does not build or print $want)" "$(tail -3 "$tmp/libc.log")"
    elif [ "$plain" = 1 ] && [ ${#added[@]} != 2 ]; then
        fail "libc (added ${added[*]}: more than -isystem $ours where nvcc compiles as it is)"
    else
        pass "libc (without nvcc-host the declarations disagree; the helper adds ${added[*]})"
    fi
fi

# ---- a default host compiler that CUDA 12.9.2 refuses
gcc=$(command -v gcc) || gcc=
if [ -z "$gcc" ] || ! compgen -c g++- | grep -qE '^g\+\+-[0-9]+$'; then
    echo "SKIP  compiler (no gcc, or no g++-N in PATH)"
else
    mkdir -p "$tmp/bin"
    printf '#!/bin/sh\ncase " $* " in *" -dumpfullversion "*|*" -dumpversion "*) echo 99.0.0; exit 0 ;; esac\nexec %s -U__GNUC__ -D__GNUC__=99 "$@"\n' "$gcc" > "$tmp/bin/gcc"
    chmod +x "$tmp/bin/gcc"
    PATH=$tmp/bin:$PATH args compiler "$nvcc12"
    if [ ${#added[@]} -lt 2 ] || [ "${added[0]}" != -ccbin ] || ! [[ ${added[1]##*/} =~ ^g\+\+-[0-9]+$ ]]; then
        fail "compiler (added ${added[*]}, not -ccbin g++-N first)" "$(cat "$tmp/compiler.said")"
    elif PATH=$tmp/bin:$PATH build compiler-control "$nvcc12"; then
        fail "compiler (the control compiles with the default host compiler)"
    elif ! PATH=$tmp/bin:$PATH build compiler "$nvcc12" "${added[@]}"; then
        fail "compiler (added ${added[*]}, and the program does not build or print $want)" "$(tail -3 "$tmp/compiler.log")"
    elif [ "$plain" = 1 ] && [ ${#added[@]} != 2 ]; then
        fail "compiler (added ${added[*]}: more than a host compiler where nvcc compiles as it is)"
    else
        pass "compiler (GCC 99 is refused; the helper adds ${added[*]})"
    fi
fi
exit $failed
