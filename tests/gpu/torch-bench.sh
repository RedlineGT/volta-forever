#!/bin/bash
# Times of PyTorch operations through the layer against a PyTorch that has code for the GPU, on a
# machine with a V100:
#   tests/gpu/torch-bench.sh [GPU [PATTERN...]]
# Runs tests/gpu/torch-bench.py twice on the same GPU: with the stock wheel of tests/gpu/torch.sh
# (2.14.0+cu130: machine code for sm_75 and later, translated by the layer), and with the last
# wheel that has sm_70 code, the same version built for CUDA 12.6, without the layer:
#   python3 -m pip install --target DIR --index-url https://download.pytorch.org/whl/cu126 \
#       --extra-index-url https://pypi.org/simple torch==2.14.0
# in $VF_TORCH_NATIVE (default $VF_TEST_ROOT/torch-cu126/site). Prints, for each operation, the
# two times of a call and their ratio, and says when the two did not compute the same thing. The
# forwarding libraries are put in front of the wheel's cuBLAS when the map is at
# $VF_OUT/torch-audit.map (or $VF_AUDIT_MAP_FILE). The stock wheel is $VF_TORCH (default
# $VF_TEST_ROOT/torch-cu130/site).
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
out=${VF_OUT:-$VF_TEST_ROOT}
layer=${VF_TORCH:-$VF_TEST_ROOT/torch-cu130/site}
native=${VF_TORCH_NATIVE:-$VF_TEST_ROOT/torch-cu126/site}
gpu=${1:-0}
[ $# -gt 0 ] && shift
for site in "$layer" "$native"; do
    [ -d "$site/torch" ] || { echo "missing: PyTorch in $site (the head of $0 says how to put it there)"; exit 2; }
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu
. "$W/tests/gpu/cache.sh"
mkdir -p "$out/logs"
proxy=${VF_PROXY:-$W/dist/lib}
map=$(vf_torch_map "$out")

PYTHONPATH=$native python3 "$W/tests/gpu/torch-bench.py" "$out/logs/bench-native.tsv" "$@" >"$out/logs/bench-native.log" 2>&1 \
    || { echo "the wheel with sm_70 code failed: $out/logs/bench-native.log"; exit 2; }
if [ -f "$map" ]; then
    LD_AUDIT=$proxy/libvf_audit.so VF_AUDIT_MAP_FILE=$map LD_LIBRARY_PATH=$proxy PYTHONPATH=$layer \
        python3 "$W/tests/gpu/torch-bench.py" "$out/logs/bench-layer.tsv" "$@" >"$out/logs/bench-layer.log" 2>&1
else
    LD_LIBRARY_PATH=$proxy PYTHONPATH=$layer python3 "$W/tests/gpu/torch-bench.py" "$out/logs/bench-layer.tsv" "$@" >"$out/logs/bench-layer.log" 2>&1
fi || { echo "the stock wheel through the layer failed: $out/logs/bench-layer.log"; exit 2; }

python3 - "$out/logs/bench-native.tsv" "$out/logs/bench-layer.tsv" <<'PY'
import math, sys
def load(path):
    return {f[0]: f[1:] for f in (line.rstrip('\n').split('\t') for line in open(path))}
native, layer = load(sys.argv[1]), load(sys.argv[2])
ratios = []
print('%-30s %10s %10s %7s' % ('operation', 'sm_70 ms', 'layer ms', 'ratio'))
for name, (ms, calls, *value) in native.items():
    other = layer.get(name)
    if other is None:
        continue
    if ms == 'failed' or other[0] == 'failed':
        print('%-30s %10s %10s         %s' % (name, ms, other[0], (other[2] if other[0] == 'failed' else value[0])[:70]))
        continue
    a, b = float(ms), float(other[0])
    # The sum of the result and the sum of its magnitudes, from each installation.
    x, mx, y, my = float(value[0]), float(value[1]), float(other[2]), float(other[3])
    # Random numbers are not the same from one run to the next: nothing to compare.
    random = name in ('randn', 'rand', 'randint', 'bernoulli', 'multinomial', 'dropout')
    # The same: the magnitudes within 2 parts in a thousand, and the sums within as much of the
    # magnitudes (of themselves would be asking a sum that cancels out for its rounding).
    scale = 2e-3 * max(mx, my)
    same = random or (x == y and mx == my) or (abs(mx - my) <= scale and abs(x - y) <= scale) or (math.isnan(x) and math.isnan(y))
    ratios.append(b / a)
    print('%-30s %10.3f %10.3f %7.2f%s' % (name, a, b, b / a, '' if same else '   NOT THE SAME: sums %g and %g, of magnitudes %g and %g' % (x, y, mx, my)))
if ratios:
    ratios.sort()
    print('%d operations: the layer takes from %.2f to %.2f times the time of the sm_70 build, median %.2f, geometric mean %.2f' % (
        len(ratios), ratios[0], ratios[-1], ratios[len(ratios) // 2], math.exp(sum(map(math.log, ratios)) / len(ratios))))
PY
