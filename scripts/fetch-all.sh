#!/bin/bash
# Fetch what development and the tests read that is not in the repository. Re-runnable: verified
# archives are not downloaded again.
#
# usage: fetch-all.sh STEP...
#   redist    CUDA 12.9.2 and 13.5.0 from NVIDIA's redistributable archives, as toolkit trees
#             (toolkits/cuda-*: compilers, disassembler, libraries; 20 GB with what follows)
#   cudnn     cuDNN for CUDA 12 and 13, and the last cuDNN with code for Volta (toolkits/cudnn-*)
#   denvdis   the instruction descriptions that reference/ reads (src/denvdis, a pinned commit)
#   demos     the official llama.cpp build for CUDA 13.5, two small models, and the CUDA libraries
#             of the last PyTorch wheel with code for sm_70 (demos/)
#   src       denvdis, and sources read or built by some tests (llama.cpp b12418, NVIDIA's samples
#             and open kernel modules, CUTLASS, ZLUDA, CuAssembler) (src/)
#   drivers   NVIDIA's driver packages, unpacked and not installed (driver/)
#   docs      a mirror of the parts of NVIDIA's documentation that the work leaned on (nvidia-docs/)
#   all       every step
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
S=$W/scripts
NV=https://developer.download.nvidia.com/compute
CUDA_OLD=12.9.2
CUDA_NEW=13.5.0
CUDNN=9.28.1
CUDNN_VOLTA=9.10.2          # the last cuDNN with engines for Volta (its CUDA 12 build)
DRIVER_NODE=580.178.04      # what the V100 nodes run
DRIVER_LAST580=580.196.07   # newest of the last branch with Volta support
DRIVER_NEW=620.70.06        # newest branch
# Not needed for the study (profilers, IDE plugins, network/fabric tools, bundled driver).
SKIP='^(nsight_|cuda_nsight|cuda_nvvp|cuda_gdb|collectx|mft|fabricmanager|imex|nvlsm|libnvsdm|libnvidia_nscq|nvidia_fs|nvidia_driver)'

redist() {
    mkdir -p "$W/dl/manifests"
    for v in $CUDA_OLD $CUDA_NEW; do
        m=$W/dl/manifests/redistrib_$v.json
        [ -s "$m" ] || curl -fsS -o "$m" "$NV/cuda/redist/redistrib_$v.json"
        python3 -I "$S/fetch-redist.py" "$NV/cuda/redist" "$m" "$W/dl/cuda-$v" "$W/toolkits/cuda-$v" --skip "$SKIP"
        ln -sfn lib "$W/toolkits/cuda-$v/lib64"   # where nvcc and CMake look for the libraries
    done
}

cudnn() {
    mkdir -p "$W/dl/manifests"
    m=$W/dl/manifests/cudnn_redistrib_$CUDNN.json
    [ -s "$m" ] || curl -fsS -o "$m" "$NV/cudnn/redist/redistrib_$CUDNN.json"
    for c in cuda12 cuda13; do
        python3 -I "$S/fetch-redist.py" "$NV/cudnn/redist" "$m" "$W/dl/cudnn-$CUDNN" "$W/toolkits/cudnn-$CUDNN-$c" --variant $c
    done
    # What stands in for a newer cuDNN on a V100: the last cuDNN with machine code for Volta.
    m=$W/dl/manifests/cudnn_redistrib_$CUDNN_VOLTA.json
    [ -s "$m" ] || curl -fsS -o "$m" "$NV/cudnn/redist/redistrib_$CUDNN_VOLTA.json"
    python3 -I "$S/fetch-redist.py" "$NV/cudnn/redist" "$m" "$W/dl/cudnn-$CUDNN_VOLTA" "$W/toolkits/cudnn-$CUDNN_VOLTA-cuda12" --variant cuda12
}

drivers() {
    mkdir -p "$W/dl/driver"
    for v in $DRIVER_NODE $DRIVER_LAST580 $DRIVER_NEW; do
        f=$W/dl/driver/NVIDIA-Linux-x86_64-$v.run
        [ -s "$f" ] || aria2c -q -x 8 -s 8 -d "$W/dl/driver" -o "$(basename "$f")" \
            "https://download.nvidia.com/XFree86/Linux-x86_64/$v/NVIDIA-Linux-x86_64-$v.run"
        # makeself archive: unpack only, nothing is installed
        [ -d "$W/driver/$v" ] || sh "$f" --extract-only --target "$W/driver/$v" >/dev/null
        curl -fsS -o "$W/driver/$v.README.txt" \
            "https://download.nvidia.com/XFree86/Linux-x86_64/$v/README/supportedchips.html" || true
    done
}

docs() {
    # Sections that describe the toolchain, binary formats, driver/runtime APIs and library APIs.
    local common="cuda-toolkit-release-notes parallel-thread-execution cuda-compiler-driver-nvcc
        cuda-binary-utilities nvvm-ir-spec libnvvm-api nvrtc nvjitlink nvfatbin ptx-compiler-api
        ptx-writers-guide-to-interoperability inline-ptx-assembly libdevice-users-guide
        cuda-features-archive turing-compatibility-guide cuda-installation-guide-linux
        cublas cufft cusparse cusolver curand nvjpeg cuda-compile-time-advisor
        cuda-driver-api cuda-runtime-api"
    fetch() {  # fetch DEST URL_BASE SECTION...
        local dest=$1 base=$2; shift 2
        mkdir -p "$dest"
        for s in "$@"; do
            ( cd "$dest" && wget -q -e robots=off -r -l 3 -np -nH --cut-dirs="$CUT" -A html,htm,pdf \
                --wait=0.1 --tries=3 --timeout=30 "$base/$s/index.html" ) || echo "docs: $base/$s incomplete" >&2
        done
    }
    CUT=1 fetch "$W/nvidia-docs/cuda-$CUDA_NEW" https://docs.nvidia.com/cuda $common cuda-programming-guide tile-ir &
    CUT=3 fetch "$W/nvidia-docs/cuda-$CUDA_OLD" https://docs.nvidia.com/cuda/archive/$CUDA_OLD $common cuda-c-programming-guide &
    wait
    mkdir -p "$W/nvidia-docs/deploy" && ( cd "$W/nvidia-docs/deploy" && wget -q -e robots=off -r -l 2 -np -nH --cut-dirs=1 -A html \
        https://docs.nvidia.com/deploy/cuda-compatibility/index.html ) || true
    python3 -I "$S/html2txt.py" "$W/nvidia-docs"
    # The fragment layouts of ldmatrix and movmatrix exist only as figures.
    for f in mma-ldmatrix-fragments mma-movmatrix-fragments-src mma-movmatrix-fragments-dst; do
        mkdir -p "$W/nvidia-docs/cuda-$CUDA_OLD/parallel-thread-execution/_images"
        curl -fsS -o "$W/nvidia-docs/cuda-$CUDA_OLD/parallel-thread-execution/_images/$f.png" \
            "https://docs.nvidia.com/cuda/archive/$CUDA_OLD/parallel-thread-execution/_images/$f.png" || true
    done
}

src() {
    mkdir -p "$W/src" && cd "$W/src"
    clone() {  # clone NAME URL [REF]
        [ -d "$1" ] && return 0
        git clone -q --depth 1 ${3:+--branch "$3"} "$2" "$1" || echo "src: $2 failed" >&2
    }
    denvdis
    cd "$W/src"
    clone ZLUDA https://github.com/vosen/ZLUDA.git
    clone CuAssembler https://github.com/cloudcores/CuAssembler.git
    clone cuda-samples https://github.com/NVIDIA/cuda-samples.git
    clone open-gpu-kernel-modules-$DRIVER_NODE https://github.com/NVIDIA/open-gpu-kernel-modules.git $DRIVER_NODE
    clone cutlass https://github.com/NVIDIA/cutlass.git
    clone llama.cpp-b12418 https://github.com/ggml-org/llama.cpp.git b12418   # source of the demonstrator build
}

denvdis() {
    # The descriptions of the instruction sets (data/sm70_*.txt, sm75_*.txt and data12/sm80_1.txt): at the commit that the
    # tables of crates/vf-tables were generated from, so that reference/generate.py gives them back.
    local commit=3769cb8a628fb9343d5466b3256c418537f7bb2a
    mkdir -p "$W/src/denvdis" && cd "$W/src/denvdis"
    [ -d .git ] || git init -q
    [ "$(git rev-parse -q --verify HEAD 2>/dev/null)" = "$commit" ] && return 0
    git fetch -q --depth 1 https://github.com/redplait/denvdis.git "$commit" && git checkout -q FETCH_HEAD \
        || echo "denvdis: fetch failed" >&2
}

demos() {
    # Official llama.cpp build for CUDA 13.5 and its CUDA runtime bundle (SHA-256 from the release page).
    local tag=b12418 base=https://github.com/ggml-org/llama.cpp/releases/download
    mkdir -p "$W/dl/llamacpp" "$W/demos/llamacpp/$tag"
    while read -r sum name dir; do
        f=$W/dl/llamacpp/$name
        [ -s "$f" ] || aria2c -q -x 8 -s 8 -d "$W/dl/llamacpp" -o "$name" "$base/$tag/$name"
        echo "$sum  $f" | sha256sum -c --quiet
        mkdir -p "$W/demos/llamacpp/$tag/$dir" && tar -xzf "$f" -C "$W/demos/llamacpp/$tag/$dir"
    done <<EOF
0efb6524d6a8078f07cf8c44c335e8b9b4a1e878dcdcedcec0f477479992e5bb llama-$tag-bin-ubuntu-cuda-13.5-x64.tar.gz llama
3712fd1bcdce682dc46e6db9e8aa8a5a61cb8d94c6fb56330eba27f3bdef1767 cudart-llama-$tag-bin-ubuntu-cuda-13.5-x64.tar.gz cudart
EOF
    # Small models for tests/gpu/llamacpp.sh: they fit on a GPU that another job fills almost entirely.
    mkdir -p "$W/demos/models"
    while read -r sum name; do
        f=$W/demos/models/$name
        [ -s "$f" ] || curl -fsSL -o "$f" "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/$name"
        echo "$sum  $f" | sha256sum -c --quiet
    done <<EOF
7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed qwen2.5-0.5b-instruct-q4_0.gguf
ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e qwen2.5-0.5b-instruct-q8_0.gguf
EOF
    # The CUDA libraries of the official PyTorch 2.14.0 wheel for CUDA 12.6, the last that has sm_70
    # kernels (SHA-256 from the index of download.pytorch.org).
    local wheel=torch-2.14.0+cu126-cp312-cp312-manylinux_2_28_x86_64.whl
    mkdir -p "$W/dl/torch" "$W/demos/torch/2.14.0+cu126"
    [ -s "$W/dl/torch/$wheel" ] || curl -fsSL -o "$W/dl/torch/$wheel" "https://download.pytorch.org/whl/cu126/${wheel//+/%2B}"
    echo "e9922441cb2ed269742a46246840c53634ca963f8b0a38d30182c28b4e8388da  $W/dl/torch/$wheel" | sha256sum -c --quiet
    unzip -o -q -j "$W/dl/torch/$wheel" torch/lib/libtorch_cuda.so torch/lib/libtorch_cuda_linalg.so torch/version.py \
        -d "$W/demos/torch/2.14.0+cu126"
}

[ $# -gt 0 ] || { sed -n '2,/^set -euo/p' "$0" | grep '^#' | cut -c3-; exit 2; }
[ "$*" = all ] && set -- redist cudnn drivers docs src demos
for step in "$@"; do
    case $step in redist|cudnn|drivers|docs|src|demos|denvdis) ;; *) echo "no such step: $step" >&2; exit 2 ;; esac
    echo "== $step $(date +%T)"
    $step
done
echo "== done $(date +%T)"
