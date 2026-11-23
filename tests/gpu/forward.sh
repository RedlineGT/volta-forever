#!/bin/bash
# GPU tests of the forwarding libraries (dist/forward, from scripts/build-forwarders.sh), on a
# machine with a V100:
#   tests/gpu/forward.sh [GPU]
# Each program of tests/gpu/forward/ is built against the CUDA 13.5 headers and libraries, then run with
# the proxy, the forwarders and the libraries of CUDA 12.9.2 in the library path (the CUDA 13.5 ones
# behind them: only cudart is taken from there). It calls a few functions of one library and checks the
# results against the host. Prints PASS or FAIL per library; exit status 1 on any FAIL.
# Builds into $VF_OUT (default $VF_TEST_ROOT).
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
T12=$W/toolkits/cuda-12.9.2
T13=$W/toolkits/cuda-13.5.0
out=${VF_OUT:-$VF_TEST_ROOT}
proxy=${VF_PROXY:-$W/dist/lib}
fwd=$W/dist/forward
src=$W/tests/gpu/forward
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${1:-0}
. "$W/tests/gpu/cache.sh"
[ -e "$fwd/libcublas.so.13" ] || { echo "missing: $fwd (run scripts/build-forwarders.sh)"; exit 2; }
mkdir -p "$out/bin" "$out/logs"

failed=0
build() {  # build NAME LIBS... : compile tests/gpu/forward/NAME.c against the CUDA 13.5 headers and libraries
    local name=$1; shift
    gcc -O2 -Wall -Wno-deprecated-declarations -I"$T13/include" -o "$out/bin/fwd_$name" "$src/$name.c" \
        -L"$T13/lib" -Wl,-rpath-link,"$T13/lib" -L"$proxy" "$@" -lcudart -lm || { echo "FAIL  build of $name"; failed=1; return 1; }
}
runcase() {  # runcase NAME : run the program with the forwarders; the output is in $log
    local name=$1
    log=$out/logs/forward-$name.log
    LD_LIBRARY_PATH=$proxy:$fwd:$T12/lib:$T13/lib VF_LOG_LEVEL=2 "$out/bin/fwd_$name" >"$log" 2>&1
    status=$?
}
verdict() {  # verdict LIBRARY... : PASS for each library the program printed "ok <library>" for, if it exited with 0
    for lib in "$@"; do
        if [ "$status" = 0 ] && grep -qx "ok $lib" "$log"; then echo "PASS  $lib"; else echo "FAIL  $lib (exit status $status; $log)"; failed=1; fi
    done
}
one() {  # one NAME "LIBS" LIBRARY... : build, run, verdict
    local name=$1 libs=$2; shift 2
    # shellcheck disable=SC2086
    if build "$name" $libs; then runcase "$name"; verdict "$@"; else for l in "$@"; do echo "FAIL  $l (not built)"; done; fi
}

one cufft "-lcufft" cufft
one cufftw "-lcufftw -lcufft" cufftw
one curand "-lcurand" curand
one cusparse "-lcusparse" cusparse
one cusolver "-lcusolver -lcublas" cusolver
one cusolvermg "-lcusolverMg" cusolvermg
one cublaslt "-lcublasLt" cublaslt
one npp "-lnppc -lnppisu -lnppidei -lnppial -lnppif -lnppim -lnppitc -lnppicc -lnppig -lnppist -lnpps" \
    nppc nppisu nppidei nppial nppif nppim nppitc nppicc nppig nppist npps npp-status
one nvjpeg "-lnvjpeg" nvjpeg
one nvrtc "-lnvrtc -l:libcuda.so.1" nvrtc

# The same programs under eager loading (CUDA_MODULE_LOADING=EAGER): the CUDA 12 runtime inside
# each 12.9 library then takes every kernel of the library at its first call, and gives up at a
# lookup that fails. cuBLASLt, cuSPARSE and cuSOLVER (through cuBLAS) have images without code for
# the V100, some with kernels that the layer refuses; until 2026-11-23 their handles could not be
# created under eager loading.
for name in cufft curand cusparse cusolver cublaslt; do
    log=$out/logs/forward-$name-eager.log
    CUDA_MODULE_LOADING=EAGER LD_LIBRARY_PATH=$proxy:$fwd:$T12/lib:$T13/lib VF_LOG_LEVEL=2 "$out/bin/fwd_$name" >"$log" 2>&1
    status=$?
    if [ "$status" = 0 ] && grep -qx "ok $name" "$log"; then echo "PASS  $name under eager loading"
    else echo "FAIL  $name under eager loading (exit status $status; $log)"; failed=1; fi
done

# First calls from 32 threads at once, in 50 processes: the 12.9 libraries are loaded once each.
if gcc -O2 -Wall -I"$T13/include" -o "$out/bin/fwd_threads" "$src/threads.c" -L"$T13/lib" -Wl,-rpath-link,"$T13/lib" -lcublas -lcufft -lcurand -lpthread; then
    bad=0
    for i in $(seq 50); do
        LD_LIBRARY_PATH=$fwd:$T12/lib:$T13/lib "$out/bin/fwd_threads" >"$out/logs/forward-threads.log" 2>&1 || { bad=$((bad + 1)); cp "$out/logs/forward-threads.log" "$out/logs/forward-threads-failed.log"; }
    done
    if [ "$bad" = 0 ]; then echo "PASS  first calls from 32 threads at once, 50 times: one load of each 12.9 library, every answer right"; else echo "FAIL  threads: $bad of 50 runs failed ($out/logs/forward-threads-failed.log)"; failed=1; fi
else
    echo "FAIL  build of threads"; failed=1
fi

# Functions CUDA 12.9 does not have: the library's "not supported" status, and one line on standard
# error, once however many calls.
one unsupported "-lcublas -lcusparse -lnvrtc" cublas-stub cusparse-stub nvrtc-stub
for want in "cublas-stub:\[volta-forever\] libcublas.so.13: cublasGetEmulationSpecialValuesSupport is not supported" \
            "cusparse-stub:\[volta-forever\] libcusparse.so.12: cusparseSpGEAM_createDescr is not supported" \
            "nvrtc-stub:\[volta-forever\] libnvrtc.so.13: nvrtcGetTileIRSize is not supported"; do
    name=${want%%:*} pattern=${want#*:}
    n=$(grep -c -- "$pattern" "$out/logs/forward-unsupported.log")
    if [ "$n" = 1 ]; then echo "PASS  $name says so once on standard error"; else echo "FAIL  $name: the line is printed $n times ($out/logs/forward-unsupported.log)"; failed=1; fi
done

# The same transform with the libraries given by the audit library's map instead of the library path:
# the forwarder asks for libcufft.so.11 and the map says where it is.
map=$out/forward-audit.map
printf 'libcufft.so.12=%s/libcufft.so.12\nlibcufft.so.11=%s/lib/libcufft.so.11\n' "$fwd" "$T12" >"$map"
log=$out/logs/forward-cufft-audit-map.log
LD_AUDIT=$proxy/libvf_audit.so VF_AUDIT_MAP_FILE=$map LD_LIBRARY_PATH=$proxy:$T13/lib VF_LOG_LEVEL=2 VF_FWD_VERBOSE=1 "$out/bin/fwd_cufft" >"$log" 2>&1
status=$?
if [ "$status" = 0 ] && grep -qx "ok cufft" "$log" && grep -q "cuda-12.9.2/lib/libcufft.so.11 is told the real compute capability" "$log"; then
    echo "PASS  cufft with the libraries given by the audit map"
else
    echo "FAIL  cufft with the libraries given by the audit map (exit status $status; $log)"; failed=1
fi
exit $failed
