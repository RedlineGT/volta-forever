#!/bin/bash
# SGLang through the layer on a machine with a V100:
#   tests/gpu/sglang.sh [GPU [ATTENTION_BACKEND [STAGE...]]]
#
# Stages (all of them by default), each PASS, FAIL or REFUSED; the exit status is 1 unless all pass:
#   a   transformers in float32 on the GPU: every token the CPU's (plain greedy)
#   b   a kernel compiled by Triton: the CPU's result, bit for bit
#   c   SGLang's engine, greedy, float16: every token the CPU's (float32, plain greedy)
#   d   SGLang's HTTP server (its /generate): the continuation of each prompt, greedy: the CPU's text
# REFUSED: the layer did not give a kernel to the program, and says which and why.
#
# ATTENTION_BACKEND: triton (default here), torch_native, flashinfer (SGLang's own default).
#   With flashinfer, c and d are refused: its attention kernels have matrix products and name
#   every register of the machine (`LIMITS.md`). The other two do not use those kernels
#   (FlashInfer still gives the normalisation, the rotary embedding and the sampling).
#
# Requires, either
#   $VF_SGLANG_VENV   a virtual environment in which SGLang was installed the usual way
#                        (`python3 -m venv DIR; DIR/bin/pip install sglang==0.5.23`, 11 GB; made on
#                        another machine with the same Python at /usr/bin, it works here through
#                        its path here). Then none of the four directories below is used.
# or (an installation put together by hand, which lacks packages that SGLang declares: its server
# does not start):
#   $VF_SGLANG_SITE   SGLang 0.5.23, flashinfer 0.7.2, deps (default $VF_TEST_ROOT/sglang/site)
#   $VF_TORCH          PyTorch 2.14.0+cu130      (default $VF_TEST_ROOT/torch-cu130/site)
#   $VF_MODEL          Qwen2.5-0.5B-Instruct     (default $VF_TEST_ROOT/models/Qwen2.5-0.5B-Instruct)
#   $VF_CUTLASS_PREFIX  the CUTLASS DSL, assembled  (default $VF_TEST_ROOT/cutlass-prefix)
# and in both cases
#   $VF_PYTHON_DEV     python3-dev headers        (default $VF_TEST_ROOT/python-dev)
#   $VF_SERVER_WAIT    seconds to wait for the HTTP server to answer (default 1800: a first start
#                        reads gigabytes of files and compiles kernels)
#   $VF_ENGINE_WAIT    seconds an engine stage may take (default 1800, for the same reason; a
#                        stage that runs out of it FAILS, and says it ran out of time)
#   $VF_PROXY          proxy lib directory        (default: dist/lib of this tree)
#   $VF_AUDIT_MAP_FILE audit map (optional)
#   The CUDA 13.5 toolkit at this tree's toolkits/cuda-13.5.0 (or $CUDA_HOME).
#
# The CUTLASS DSL: SGLang imports `cutlass` when it loads a model, on any GPU. With `pip --target`
# the .pth file by which nvidia-cutlass-dsl puts itself on the path is not processed, so the
# package was assembled by hand (FlashInfer's Python API, the DSL's MLIR backend, and an alias
# that FlashInfer 0.7.2 lacks). An installation made the usual way needs none of this.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
host=$(hostname)
out=${VF_OUT:-$VF_TEST_ROOT}
gpu=${1:-0}
backend=${2:-${VF_ATTENTION_BACKEND:-triton}}
[ $# -gt 0 ] && shift
[ $# -gt 0 ] && shift
stages=" ${*:-a b c d} "
want() { [[ "$stages" == *" $1 "* ]]; }

venv=${VF_SGLANG_VENV:-}
site=${VF_SGLANG_SITE:-$VF_TEST_ROOT/sglang/site}
torch_site=${VF_TORCH:-$VF_TEST_ROOT/torch-cu130/site}
model=${VF_MODEL:-$VF_TEST_ROOT/models/Qwen2.5-0.5B-Instruct}
cutlass=${VF_CUTLASS_PREFIX:-$VF_TEST_ROOT/cutlass-prefix}
pydev=${VF_PYTHON_DEV:-$VF_TEST_ROOT/python-dev}
proxy=${VF_PROXY:-$W/dist/lib}
map=$(vf_torch_map "$out")

if [ -n "$venv" ]; then
    [ -x "$venv/bin/python3" ] || { echo "missing: $venv/bin/python3 (the virtual environment)"; exit 2; }
    py=("$venv/bin/python3")
else
    [ -d "$site/sglang" ]       || { echo "missing: $site/sglang (SGLang 0.5.23 and what it needs)"; exit 2; }
    [ -d "$torch_site/torch" ]   || { echo "missing: $torch_site/torch (PyTorch)"; exit 2; }
    [ -d "$cutlass/cutlass" ]    || { echo "missing: $cutlass/cutlass (the CUTLASS DSL: see the head of $0)"; exit 2; }
    py=(python3 -s)
fi
[ -d "$model" ]              || { echo "missing: $model (the model)"; exit 2; }
[ -f "$proxy/libcuda.so.1" ] || { echo "missing: $proxy/libcuda.so.1 (build with scripts/build.sh)"; exit 2; }

# --- Caches and temporary files under the output directory, not under /root ---
cache=$out/sglang-cache
export HOME=$cache
export TMPDIR=$cache/tmp
export XDG_CACHE_HOME=$cache/.cache
export TRITON_CACHE_DIR=$cache/triton
export CUDA_CACHE_PATH=$cache/nv-cache            # (the driver's compiled kernels: ~/.nv otherwise)
export FLASHINFER_WORKSPACE_BASE=$cache
export SGLANG_CACHE_DIR=$cache/sglang
export SGLANG_JIT_CACHE_DIR=$cache/sglang-jit
export HF_HOME=$cache/hf
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$SGLANG_CACHE_DIR" "$SGLANG_JIT_CACHE_DIR" "$HF_HOME"

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu
source "$W/tests/gpu/cache.sh"
# What this run alone writes (the CPU's references, the server's log, the stamp): a directory of
# its own, so that two runs at once do not write each other's files. Removed at the end, the
# server's log kept when a stage did not pass.
run=$(mktemp -d "$cache/run-XXXXXXXX") || exit 1
stamp=$run/started
touch "$stamp"
failed=0
engine_wait=${VF_ENGINE_WAIT:-1800}
cleanup() {
    vf_cleanup_proxy_cache
    if [ "$1" = 0 ] && [ "$failed" = 0 ]; then rm -rf "$run"; else echo "kept: $run"; fi
}
trap 'cleanup "$?"' EXIT
export PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1

[ -n "$venv" ] || export PYTHONPATH="$cutlass:$site:$torch_site"
export CUDA_HOME=${CUDA_HOME:-$W/toolkits/cuda-13.5.0}
[ -d "$pydev/usr/include/python3.12" ] && export CPATH="$pydev/usr/include/python3.12:$pydev/usr/include"
export LIBRARY_PATH="$proxy:${LIBRARY_PATH:-}"
vf_layer "$proxy" "$map"
export VF_LOG_LEVEL=${VF_LOG_LEVEL:-1}

stop_server() {
    # The server and the processes it started: asked to stop, then stopped if they have not
    # ended after half a minute (an engine reading its files from a slow disk does not hear).
    local pid=$1 children
    children=$(pgrep -P "$pid" 2>/dev/null)
    kill "$pid" 2>/dev/null
    for _ in $(seq 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" $children 2>/dev/null
    wait "$pid" 2>/dev/null
}
refused() {
    # What the layer says when it does not give a kernel to the program.
    echo "$1" | grep -m1 'cannot run on this GPU' | cut -c1-300
}
verdict() {
    # verdict STAGE OUTPUT WORD: PASS when the program's verdict is WORD, else REFUSED or FAIL. The
    # verdict is what follows `verdict: ` on the last line of the output that starts so: what
    # else the program, its libraries and its kernels printed is not looked at for it. With
    # ALL_MATCH for WORD, a text that leaves the CPU's only where the CPU itself had a near tie
    # (greedy_compare.py) passes too, and says so.
    local said
    said=$(echo "$2" | sed -n 's/^verdict: //p' | tail -1)
    if [ "$said" = "$3" ]; then
        echo "PASS stage-$1"
    elif [ "$3" = ALL_MATCH ] && [[ "$said" =~ ^MATCH_BUT_NEAR_TIES\ [0-9]+$ ]]; then
        echo "PASS stage-$1 (with ${said#MATCH_BUT_NEAR_TIES } near tie(s): see the lines above)"
    elif [ -n "$(refused "$2")" ]; then
        echo "REFUSED stage-$1: $(refused "$2")"; failed=1
    elif echo "$2" | grep -q '^RAN OUT OF TIME$'; then
        echo "FAIL stage-$1: ran out of time (\$VF_ENGINE_WAIT is $engine_wait s)"; failed=1
    else
        echo "FAIL stage-$1"; echo "$2" | tail -30; failed=1
    fi
}

echo "attention backend: $backend; $("${py[@]}" -c 'import importlib.metadata as m; print("SGLang", m.version("sglang"), "PyTorch", m.version("torch"), "FlashInfer", m.version("flashinfer-python"))' 2>/dev/null || echo "versions not read")"
ref=$run/ref-f32.json
if want a || want c || want d; then
    echo "=== the CPU's reference (transformers, float32, 50 tokens, plain greedy) ==="
    CUDA_VISIBLE_DEVICES="" MODEL="$model" DTYPE=float32 MAX_TOKENS=50 OUTFILE="$ref" \
        "${py[@]}" "$W/tests/gpu/vllm-ref.py" || { echo "FAIL: the CPU's reference"; exit 1; }
fi

# ---------- stage a: transformers on the GPU against the CPU ----------
if want a; then
    echo ""
    echo "=== stage a: transformers, float32, on the GPU through the layer ==="
    gpu_out=$(MODEL="$model" DTYPE=float32 MAX_TOKENS=50 REF_FILE="$ref" "${py[@]}" "$W/tests/gpu/vllm-gpu.py" 2>&1) || true
    echo "$gpu_out" | grep -E "prompt [0-9]|tok/s|ALL_MATCH|DIFFER"
    verdict a "$gpu_out" ALL_MATCH
fi

# ---------- stage b: a kernel compiled by Triton ----------
if want b; then
    echo ""
    echo "=== stage b: a kernel compiled by Triton ==="
    if [ -n "${CPATH:-}" ]; then
        triton_out=$("${py[@]}" "$W/tests/gpu/vllm-triton.py" 2>&1) || true
        echo "$triton_out" | grep -E "^PASS|^FAIL" | cut -c1-200
        if echo "$triton_out" | grep -q "^PASS"; then
            echo "PASS stage-b"
        elif [ -n "$(refused "$triton_out")" ]; then
            echo "REFUSED stage-b: $(refused "$triton_out")"; failed=1
        else
            echo "FAIL stage-b"; echo "$triton_out" | tail -30; failed=1
        fi
    else
        echo "FAIL stage-b: no Python.h (Triton compiles a C stub: \$VF_PYTHON_DEV)"; failed=1
    fi
fi

# ---------- stage c: SGLang's engine, greedy ----------
if want c; then
    echo ""
    echo "=== stage c: SGLang's engine, greedy, float16, attention by $backend, against the CPU ==="
    gen_out=$(MODEL="$model" REF_FILE="$ref" ATTENTION_BACKEND="$backend" timeout "$engine_wait" \
        "${py[@]}" "$W/tests/gpu/sglang-generate.py" 2>&1; [ $? = 124 ] && echo "RAN OUT OF TIME") || true
    echo "$gen_out" | grep -E "prompt [0-9]|engine:|reference:|tok/s|engine created|^sglang " | cut -c1-200
    verdict c "$gen_out" ALL_MATCH
fi

# ---------- stage d: SGLang's HTTP server ----------
if want d; then
    echo ""
    echo "=== stage d: SGLang's HTTP server, greedy, float16, attention by $backend, against the CPU ==="
    port=$((20000 + RANDOM % 20000))
    "${py[@]}" -m sglang.launch_server --model-path "$model" --dtype float16 --attention-backend "$backend" \
        --mem-fraction-static 0.7 --host 127.0.0.1 --port $port > "$run/server.log" 2>&1 &
    server=$!
    up=
    for _ in $(seq ${VF_SERVER_WAIT:-1800}); do
        kill -0 $server 2>/dev/null || break
        if python3 -I -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$port/health', timeout=2)" 2>/dev/null; then
            up=1; break
        fi
        sleep 1
    done
    if [ -n "$up" ]; then
        server_out=$(PORT=$port REF_FILE="$ref" python3 -I "$W/tests/gpu/sglang-request.py" 2>&1) || true
        echo "$server_out" | cut -c1-200
        # (A refusal is in the server's log, not in what the client was answered.)
        verdict d "$server_out
$(grep 'cannot run on this GPU' "$run/server.log")" ALL_MATCH
    else
        echo "the server did not come up; the end of its log ($run/server.log):"
        verdict d "$(tail -30 "$run/server.log")
$(grep 'cannot run on this GPU' "$run/server.log")" ALL_MATCH
    fi
    stop_server $server
fi

# ---------- what the run left outside its own directories ----------
# (Files written under $HOME, /tmp or /dev/shm since the start and still there, the proxy's cache
# and the GPU locks apart. Another process of the machine may have written them: they are listed,
# and do not change the exit status. Known: SGLang leaves a `/dev/shm/sglang_loads_*.shm` of
# 16 kB at each start of an engine, and Python's multiprocessing a `sem.mp-*` when a process is
# stopped; a program run without CUDA_CACHE_PATH leaves the driver's compiled kernels under ~/.nv.)
echo ""
new=$( { find /root -xdev -type f -newer "$stamp" 2>/dev/null
         find /dev/shm -xdev -type f -newer "$stamp" 2>/dev/null
         find /tmp -xdev -type f -newer "$stamp" -not -path '/tmp/vf-cache-*' -not -name 'vf-*.lock' 2>/dev/null; } | head -20)
if [ -n "$new" ]; then
    echo "files written under /root, /tmp or /dev/shm during the run and left there:"; echo "$new"
else
    echo "nothing left under /root, /tmp or /dev/shm by the run (the proxy's cache apart)"
fi
exit $failed
