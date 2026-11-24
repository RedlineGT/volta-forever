#!/bin/bash
# vLLM through the layer on a machine with a V100:
#   tests/gpu/vllm.sh [GPU [STAGE...]]
#
# Stages (all of them by default), each PASS, FAIL or REFUSED; the exit status is 1 unless all pass:
#   a   transformers in float32 on the GPU: every token the CPU's (plain greedy)
#   b   a kernel compiled by Triton: the CPU's result, bit for bit
#   c   vLLM's engine, greedy, float16: every token the CPU's (float32, plain greedy)
#   c2  vLLM's engine, sampling (top-k 50, top-p 0.9): every token it samples is among the 50
#       most likely and within 0.9 of their probability, by the log-probabilities of the logits
#       that the engine itself computed (the kernels of the sampling; the logits are c's)
#   d   vLLM's HTTP server (the OpenAI API): a completion for each prompt, greedy with the
#       repetition penalty that the model's own configuration asks for (the server applies it by
#       default, and then its kernel for penalties runs): the CPU's text, made with that penalty
# REFUSED: the layer did not give a kernel to the program, and says which and why. (Until
# 2026-11-19, c, c2 and d were refused: a kernel of vLLM's sampling names every register of the
# machine; `docs/translator.md`, "Registers", says how it is translated.)
#
# Requires, either
#   $VF_VLLM_VENV     a virtual environment in which vLLM was installed the usual way
#                        (`python3 -m venv DIR; DIR/bin/pip install vllm==0.34.0`; made on another
#                        machine with the same Python at /usr/bin, it works here through its path
#                        here). Then none of the five directories below is used.
# or (an installation put together by hand):
#   $VF_VLLM_SITE     vLLM 0.34.0 and its deps (default $VF_TEST_ROOT/vllm/site)
#   $VF_TORCH          PyTorch 2.14.0+cu130    (default $VF_TEST_ROOT/torch-cu130/site)
#   $VF_VLLM_MODEL     Qwen2.5-0.5B-Instruct   (default $VF_TEST_ROOT/models/Qwen2.5-0.5B-Instruct)
#   $VF_VLLM_EXTRA     extra site (loguru etc.) (default $VF_TEST_ROOT/vllm-extra/site)
#   $VF_VLLM_DEPS      what the two sites above lack of what vLLM declares (torchvision, the
#                         CUTLASS DSL, jmespath; default $VF_TEST_ROOT/vllm-deps/site
#                         when it exists). Its nvidia_cutlass_dsl/dsl_packages is added to the path:
#                         that package puts itself there by a .pth file, which `pip --target` does
#                         not process. (A vLLM installed the usual way needs none of the three.)
# and in both cases
#   Python 3.12 headers at $VF_PYTHON_DEV (for Triton's C stub)
#   $VF_SERVER_WAIT    seconds to wait for the HTTP server to answer (default 1800: a first start
#                        reads gigabytes of files and compiles kernels)
#   $VF_ENGINE_WAIT    seconds an engine stage may take (default 1800, for the same reason; a
#                        stage that runs out of it FAILS, and says it ran out of time)
#   $TMPDIR            where the directory of vLLM's sockets is made (default /tmp; a local
#                        directory, with a short path), unless $VLLM_RPC_BASE_PATH is given
#   $VF_PROXY          proxy lib directory      (default: dist/lib of this tree)
#   $VF_AUDIT_MAP_FILE audit map (optional; cuBLAS 12.9.2 forwarding)
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
host=$(hostname)
out=${VF_OUT:-$VF_TEST_ROOT}
gpu=${1:-0}
[ $# -gt 0 ] && shift
stages=" ${*:-a b c c2 d} "
want() { [[ "$stages" == *" $1 "* ]]; }

venv=${VF_VLLM_VENV:-}
vllm_site=${VF_VLLM_SITE:-$VF_TEST_ROOT/vllm/site}
torch_site=${VF_TORCH:-$VF_TEST_ROOT/torch-cu130/site}
model=${VF_VLLM_MODEL:-$VF_TEST_ROOT/models/Qwen2.5-0.5B-Instruct}
extra=${VF_VLLM_EXTRA:-$VF_TEST_ROOT/vllm-extra/site}
deps=${VF_VLLM_DEPS:-$VF_TEST_ROOT/vllm-deps/site}
pydev=${VF_PYTHON_DEV:-$VF_TEST_ROOT/python-dev}
proxy=${VF_PROXY:-$W/dist/lib}
map=$(vf_torch_map "$out")

if [ -n "$venv" ]; then
    [ -x "$venv/bin/python3" ] || { echo "missing: $venv/bin/python3 (the virtual environment)"; exit 2; }
    py=("$venv/bin/python3")
else
    for d in "$vllm_site/transformers" "$torch_site/torch"; do
        [ -d "$d" ] || { echo "missing: $d (see the head of $0)"; exit 2; }
    done
    py=(python3 -s)
fi
[ -d "$model" ] || { echo "missing: $model (see the head of $0)"; exit 2; }
[ -f "$proxy/libcuda.so.1" ] || { echo "missing: $proxy/libcuda.so.1 (build with scripts/build.sh)"; exit 2; }

# --- Caches and temporary files under the output directory, not under /root ---
# (The caller's TMPDIR is kept for the directory of vLLM's sockets, below.)
caller_tmp=${TMPDIR:-/tmp}
cache=$out/vllm-cache
export HOME=$cache
export TMPDIR=$cache/tmp
export XDG_CACHE_HOME=$cache/.cache
export TRITON_CACHE_DIR=$cache/triton
export CUDA_CACHE_PATH=$cache/nv-cache            # (the driver's compiled kernels: ~/.nv otherwise)
export HF_HOME=$cache/hf
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$HF_HOME"

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
# vLLM appends a UUID to each socket name. Keep sockets on the node under
# a short path, even when the model caches live deep in an output tree.
rpc_dir=
cleanup() {
    vf_cleanup_proxy_cache
    if [ -n "$rpc_dir" ]; then rm -rf -- "$rpc_dir"; fi
    if [ "$1" = 0 ] && [ "$failed" = 0 ]; then rm -rf "$run"; else echo "kept: $run"; fi
}
trap 'cleanup "$?"' EXIT
if [ -z "${VLLM_RPC_BASE_PATH:-}" ]; then
    rpc_dir=$(mktemp -d "$caller_tmp/vf-vllm-rpc.XXXXXXXX") || exit 1
    export VLLM_RPC_BASE_PATH=$rpc_dir
fi
export PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1

pp="$vllm_site:$torch_site"
[ -d "$extra" ] && pp="$extra:$pp"
if [ -d "$deps" ]; then
    pp="$deps:$pp"
    [ -d "$deps/nvidia_cutlass_dsl/dsl_packages" ] && pp="$deps/nvidia_cutlass_dsl/dsl_packages:$pp"
fi
[ -n "$venv" ] || export PYTHONPATH="$pp"
vf_layer "$proxy" "$map"
[ -d "$pydev/usr/include/python3.12" ] && export CPATH="$pydev/usr/include/python3.12:$pydev/usr/include"
export LIBRARY_PATH="$proxy:${LIBRARY_PATH:-}"
export VF_LOG_LEVEL=1

echo "$("${py[@]}" -c 'import importlib.metadata as m; print("vLLM", m.version("vllm"), "PyTorch", m.version("torch"), "Triton", m.version("triton"))' 2>/dev/null || echo "versions not read")"
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
source "$W/tests/gpu/verdict.sh"

# ---------- the CPU's answers ----------
# (Plain greedy for a to c: what the engines do at a temperature of 0. With the penalty of the
# model's configuration for d: what vLLM's server does by default.)
ref=$run/ref-f32.json
refp=$run/ref-f32-penalty.json
if want a || want c; then
    echo "=== the CPU's reference (transformers, float32, 50 tokens, plain greedy) ==="
    CUDA_VISIBLE_DEVICES="" MODEL="$model" DTYPE=float32 MAX_TOKENS=50 OUTFILE="$ref" \
        "${py[@]}" "$W/tests/gpu/vllm-ref.py" || { echo "FAIL: the CPU's reference"; exit 1; }
fi

# ---------- stage a: transformers on the GPU against the CPU ----------
if want a; then
    echo ""
    echo "=== stage a: transformers, float32, on the GPU through the layer ==="
    gpu_out=$(MODEL="$model" DTYPE=float32 MAX_TOKENS=50 REF_FILE="$ref" "${py[@]}" "$W/tests/gpu/vllm-gpu.py" 2>&1)
    status=$?
    echo "$gpu_out" | grep -E "prompt [0-9]|tok/s|ALL_MATCH|DIFFER"
    verdict a "$gpu_out" ALL_MATCH "$status"
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

# ---------- stage c: vLLM's engine, greedy ----------
if want c; then
    echo ""
    echo "=== stage c: vLLM's engine, greedy, float16, against the CPU ==="
    vllm_out=$(MODEL="$model" DTYPE=float16 MAX_TOKENS=50 TEMPERATURE=0 REF_FILE="$ref" \
        timeout -k 60 "$engine_wait" "${py[@]}" "$W/tests/gpu/vllm-engine.py" 2>&1)
    status=$?
    echo "$vllm_out" | grep -E "prompt [0-9]|MATCH|NEAR TIE|DIFFER|tok/s|^FAIL" | cut -c1-200
    verdict c "$vllm_out" ALL_MATCH "$status"
fi

# ---------- stage c2: vLLM's engine, sampling ----------
if want c2; then
    echo ""
    echo "=== stage c2: vLLM's engine, sampling (top_k=50, top_p=0.9, seed=42) ==="
    vllm_sample=$(MODEL="$model" DTYPE=float16 MAX_TOKENS=50 TEMPERATURE=0.8 TOP_K=50 TOP_P=0.9 SEED=42 \
        timeout -k 60 "$engine_wait" "${py[@]}" "$W/tests/gpu/vllm-engine.py" 2>&1)
    status=$?
    echo "$vllm_sample" | grep -E "prompt [0-9]|WITHIN|OUTSIDE|tok/s|^FAIL" | cut -c1-200
    verdict c2 "$vllm_sample" ALL_WITHIN "$status"
fi

# ---------- stage d: vLLM's HTTP server ----------
if want d; then
    echo ""
    echo "=== stage d: vLLM's HTTP server, greedy with the model's own repetition penalty, against the CPU ==="
    CUDA_VISIBLE_DEVICES="" MODEL="$model" DTYPE=float32 MAX_TOKENS=32 PENALTY=model OUTFILE="$refp" \
        "${py[@]}" "$W/tests/gpu/vllm-ref.py" > /dev/null 2>&1 || { echo "FAIL: the CPU's reference with the penalty"; exit 1; }
    port=$((20000 + RANDOM % 20000))
    "${py[@]}" -m vllm.entrypoints.openai.api_server --model "$model" --served-model-name model --dtype float16 \
        --max-model-len 512 --gpu-memory-utilization 0.8 --enforce-eager --host 127.0.0.1 --port $port \
        > "$run/server.log" 2>&1 &
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
        server_out=$(PORT=$port REF_FILE="$refp" python3 -I "$W/tests/gpu/vllm-request.py" 2>&1)
        status=$?
        echo "$server_out" | cut -c1-200
        grep -m1 -i "default sampling parameters" "$run/server.log" | sed 's/^.*[Dd]efault sampling/the server: default sampling/' | cut -c1-220
        # (A refusal is in the server's log, not in what the client was answered.)
        verdict d "$server_out
$(grep 'cannot run on this GPU' "$run/server.log")" ALL_MATCH "$status"
    else
        echo "the server did not come up; the end of its log ($run/server.log):"
        verdict d "$(tail -30 "$run/server.log")" ALL_MATCH 1
    fi
    stop_server $server
fi

# ---------- what the run left outside its own directories ----------
# (Files written under $HOME, /tmp or /dev/shm since the start and still there, the proxy's cache
# and the GPU locks apart. Another process of the machine may have written them: they are listed,
# and do not change the exit status. Known: Python's multiprocessing leaves a `sem.mp-*` when a
# process is stopped; a program run without CUDA_CACHE_PATH leaves the driver's compiled kernels
# under ~/.nv.)
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
