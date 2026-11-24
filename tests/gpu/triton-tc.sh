#!/bin/bash
# Triton's matrix products with VF_TRITON_TC on and off, through the launcher of this tree, on a
# machine with a V100:
#   tests/gpu/triton-tc.sh [GPU]
# Runs tests/gpu/triton-tc.py in five processes that share one cache of Triton (TRITON_CACHE_DIR):
# off, on, off again, on again, on with VF_TRANSLATE=first; then the verdicts, each PASS or FAIL
# (exit status 1 unless all pass):
#   - what each reader of the capability is told is the same on as off: PyTorch 7.5, Triton 7.5,
#     NVML 7.5 (the switch changes what Triton compiles, not what it is told);
#   - each kernel computes within its tolerance of float64 on the CPU, on as off; with the switch
#     on, a kernel whose products are float16, bfloat16 or int8 is compiled for sm_80 and takes
#     less time than off; a kernel with a float32 product is compiled for sm_75 and gives the same
#     bits as off; a block whose pipelined sm_80 kernel needs more shared memory than a V100 has
#     runs (fewer stages); an int8 product, which Triton cannot compile for 7.5, runs, exactly;
#   - the cache: the second process (on) compiles its kernels for sm_80 rather than take what the
#     first (off) left, the third (off) and the fourth (on) give the bits of the first and of the
#     second; the fifth (VF_TRANSLATE=first, with which the switch does nothing) takes nothing that
#     the switch made (until 2026-11-23 it took the sm_80 cubins: "a PTX JIT compilation failed");
#   - with $VF_VLLM_VENV, vLLM's fused_moe_kernel and kernel_unified_attention in float16 and
#     bfloat16 (within tolerance, faster on); torch.compile of a matrix product with Inductor's
#     Triton template alone (max-autotune): within tolerance, on and off, and its template
#     compiled for sm_80 on (the layer's lines say so).
# Needs a Python with PyTorch 2.14 and its Triton: the one of $VF_VLLM_VENV, else python3 with
# $VF_TORCH (default $VF_TEST_ROOT/torch-cu130/site); Python 3.12 headers ($VF_PYTHON_DEV) for
# Triton's C stub; the layer built (scripts/build.sh), the forwarding libraries (dist/forward) and
# the CUDA 12.9.2 libraries where the launcher finds them (INSTALL.md).
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
out=${VF_OUT:-$VF_TEST_ROOT}/triton-tc
gpu=${1:-0}
pydev=${VF_PYTHON_DEV:-$VF_TEST_ROOT/python-dev}
rm -rf "$out"
mkdir -p "$out/home" "$out/tmp" "$out/triton" "$out/inductor"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu
export HOME=$out/home TMPDIR=$out/tmp XDG_CACHE_HOME=$out/home/.cache TRITON_CACHE_DIR=$out/triton
export TORCHINDUCTOR_CACHE_DIR=$out/inductor CUDA_CACHE_PATH=$out/home/nv PYTHONDONTWRITEBYTECODE=1
export DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1 VF_QUIET=1
[ -d "$pydev/usr/include/python3.12" ] && export CPATH="$pydev/usr/include/python3.12:$pydev/usr/include"
vllm=()
if [ -n "${VF_VLLM_VENV:-}" ]; then
    py=$VF_VLLM_VENV/bin/python3
    vllm=(--vllm)
else
    py=python3
    export PYTHONPATH=${VF_TORCH:-$VF_TEST_ROOT/torch-cu130/site}
fi
run() {  # label, VF_TRITON_TC, what...
    local label=$1 tc=$2 first=; shift 2
    [ "$label" = first ] && first=first
    VF_TC_LABEL=$label VF_TRITON_TC=$tc VF_LOG="$out/$label.vflog" VF_LOG_LEVEL=2 ${first:+VF_TRANSLATE=$first} \
        "$W/scripts/vf" "$py" "$W/tests/gpu/triton-tc.py" "$@" >>"$out/$label.jsonl" 2>>"$out/$label.err"
    echo "$label ($*): status $?"
}
# off, on, off again, on again, then on with VF_TRANSLATE=first (the layer takes machine code
# before PTX: the switch does nothing, and what it made is not taken from the cache).
for label in off on off2 on2 first; do
    case $label in on*|first) tc=1 ;; *) tc=0 ;; esac
    run "$label" "$tc" probe
    run "$label" "$tc" kernels "${vllm[@]}"
done
if [ -n "${VF_VLLM_VENV:-}" ]; then
    run off 0 compile
    run on 1 compile
fi
python3 - "$out" <<'EOF'
import json, sys
out = sys.argv[1]
def load(label):
    rows = {}
    try:
        for line in open(f'{out}/{label}.jsonl'):
            if line.startswith('{'):
                r = json.loads(line)
                rows[r.get('case') or r['what']] = r
    except OSError:
        pass
    return rows
off, on, off2, on2, first = (load(x) for x in ('off', 'on', 'off2', 'on2', 'first'))
try:
    log = open(f'{out}/on.vflog', errors='replace').read()
except OSError:
    log = ''
bad = 0
def verdict(ok, what):
    global bad
    bad += not ok
    print(('PASS' if ok else 'FAIL') + ': ' + what)
p0, p1 = off.get('probe', {}), on.get('probe', {})
verdict(all(p.get('torch') == '7.5' and p.get('triton') == 75 and p.get('nvml') == '7.5' for p in (p0, p1)),
        'told 7.5, on as off: PyTorch %s/%s, Triton %s/%s, NVML %s/%s' % (
            p0.get('torch'), p1.get('torch'), p0.get('triton'), p1.get('triton'), p0.get('nvml'), p1.get('nvml')))
for name, r1 in on.items():
    if r1.get('what') != 'kernel':
        continue
    r0 = off.get(name, {})
    # (float32 outputs of the GEMMs; vLLM's kernels write float16 or bfloat16)
    limit = 1e-2 if name.startswith('vllm') else 2e-3
    if 'error' in r1:
        verdict(False, f'{name}: on: {r1["error"]}')
        continue
    if name.startswith('i8'):
        verdict(r1.get('err') == 0 and r1.get('target') == 'sm_80',
                f'{name}: on: exact {r1.get("err") == 0}, {r1.get("target")}; off: {r0.get("error", r0.get("target"))[:120]}')
        continue
    if 'error' in r0:
        verdict(False, f'{name}: off: {r0["error"]}')
        continue
    e1, e0 = r1.get('err', 1), r0.get('err', 1)
    verdict(e1 <= limit and e0 <= limit, f'{name}: error {e1:.3g} on, {e0:.3g} off (limit {limit:g})')
    if name.startswith('f32'):
        verdict(r1.get('target') == 'sm_75' and r1['hash'] == r0['hash'], f'{name}: on {r1.get("target")}, the bits of off: {r1["hash"] == r0["hash"]}')
    else:
        if 'target' in r1:
            verdict(r1['target'] == 'sm_80' and r0.get('target') == 'sm_75', f'{name}: {r1["target"]} on, {r0.get("target")} off')
        verdict(r1['us'] < r0['us'], f'{name}: {r1["us"]:.0f} us on, {r0["us"]:.0f} us off ({r0["us"] / r1["us"]:.1f} times)')
    if name in first:
        f = first[name]
        if name.startswith('i8'):
            ok = ('error' in f) == ('error' in off.get(name, {}))   # as without the switch
        else:
            ok = 'error' not in f and f.get('target') in (None, 'sm_75') and f.get('err', 1) <= limit
        verdict(ok, f'{name}: with VF_TRANSLATE=first on the same cache: {f.get("error", "")[:80] or f.get("target", "")} '
                    f'error {f.get("err", "-")}')
    if name in off2 and name in on2:
        verdict(off2[name].get('hash') == r0.get('hash') and on2[name].get('hash') == r1.get('hash'),
                f'{name}: the cache: off again gives the bits of off, on again those of on')
if 'compile' in on:
    c1, c0 = on['compile'], off.get('compile', {})
    if 'error' in c1 or 'error' in c0:
        verdict(False, f'torch.compile: on {c1.get("error", "ok")}; off {c0.get("error", "ok")}')
    else:
        verdict(c1['err'] <= 2e-3 and c0['err'] <= 2e-3, f'torch.compile: error {c1["err"]:.3g} on, {c0["err"]:.3g} off; {c1["us"]:.0f} us on, {c0["us"]:.0f} us off')
        import re
        made = re.findall(r'Triton kernel (triton_(?:tem|mm)\w*): compiled for sm_80', log)
        verdict(bool(made), 'torch.compile: Inductor\'s template compiled for sm_80 (the layer\'s lines: %d '
                'kernels, %s)' % (len(made), ', '.join(sorted(set(made))[:3])))
print('=== triton-tc: %s ===' % ('PASS' if bad == 0 else '%d FAIL' % bad))
sys.exit(1 if bad else 0)
EOF
