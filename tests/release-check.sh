#!/bin/bash
# The checks of a release, one after the other, with a table at the end:
#   tests/release-check.sh [--full]
#
# Host part (no GPU), in this order: formatting, lints (warnings are errors), the unit tests (in an
# empty home of their own, which they must leave empty: tests/scripts/untouched-home.sh), the
# build of scripts/build.sh, every host suite that the CI runs (tests/README.md, "What the CI
# runs"), the host suites it does not run when what they need is here (jitlink_objects.py and the
# CUDA 12.9.2 toolkit, package-refusals.sh and the forwarding libraries, the bf16 kernels rebuilt
# with the CUDA 12.9.2 compiler, tests/scripts/nvcc-host.sh), and tests/sass/compare.sh on the
# test kernels and on PyTorch's cubins when `TORCH` names them. With --full, then the independent
# validator over the corpora: validate-corpus.sh on `TORCH`, heldout-validate.sh on `HELDOUT` and
# controls.sh (hours, and the corpora of tests/README.md: a step whose corpus is missing is
# skipped). Last, `executable`: every script that a step or a GPU command runs directly has the
# mode 100755 in git's index.
#
# GPU part: printed, not run: the commands of tests/gpu/ with the settings a release is checked
# with, then how to check an installed archive instead of this tree. VF_RELEASE_GPU_RUN='PREFIX'
# runs each of the tree's commands as `PREFIX 'COMMAND'` (COMMAND one argument, a shell command
# line, as a job runner takes it), one after the other, and puts them in the table (a suite for two
# GPUs, tests/gpu/multi-gpu.sh, is a skip when it ends with 2: fewer than two GPUs visible):
#   VF_RELEASE_GPU_TREE   the path of this tree where COMMAND runs (default: this path)
#   VF_GPU                the PCI-order index of the GPU there; by default CUDA_VISIBLE_DEVICES
#                         as COMMAND finds it (its first entry), else 0
#
# Every step writes its output to $VF_OUT/release-check/STEP.log (VF_OUT: tests/README.md,
# "Settings"); JOBS is given to the steps that run several processes. The exit status is 1 when a
# step failed, 2 when none failed but a step could not run (what it needs is missing), else 0.
set -uo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
. "$W/tests/gpu/env.sh"
full=0
for arg in "$@"; do
    case $arg in
        --full) full=1 ;;
        -h | --help) sed -n '2,/^set /{/^set /d;s/^# \{0,1\}//;p}' "$0"; exit 0 ;;
        *) echo "usage: $0 [--full]" >&2; exit 2 ;;
    esac
done
out=${VF_OUT:-$VF_TEST_ROOT}/release-check
rm -rf "$out" && mkdir -p "$out" || exit 2
cargo=${CARGO:-$(command -v cargo || echo "$HOME/.cargo/bin/cargo")}
TORCH=${TORCH:-$VF_TEST_ROOT/torch-cubins}
HELDOUT=${HELDOUT:-}

rows=()
failed=0 skipped=0
# The scripts of the tree that the steps and the GPU commands run directly (not through an interpreter).
invoked=()
# script_of WORD...: the first word of a command that is neither `env` nor an assignment, when it is a
# file of the tree.
script_of() {
    local word
    for word in "$@"; do
        case $word in env | *=*) continue ;; esac
        if [ -f "$W/$word" ]; then echo "$word"; fi
        return 0
    done
}
# step NAME COMMAND...: run COMMAND in the tree, its output to NAME.log; PASS when it ends with 0.
# With skip_on=N set for the call, an end with status N is a skip (what the step needs is not here:
# multi-gpu.sh's 2 for fewer than two GPUs visible), not a failure.
step() {
    local name=$1 start=$SECONDS status result script
    shift
    script=$(script_of "$@")
    [ -z "$script" ] || invoked+=("$script")
    printf '%-24s ' "$name"
    (cd "$W" && "$@") > "$out/$name.log" 2>&1
    status=$?
    if [ $status -eq 0 ]; then result=PASS
    elif [ -n "${skip_on:-}" ] && [ $status -eq "$skip_on" ]; then result=SKIP; skipped=1
    else result=FAIL; failed=1; fi
    printf '%s (%d s)\n' "$result" $((SECONDS - start))
    rows+=("$(printf '%-24s %-5s %6d s  %s' "$name" "$result" $((SECONDS - start)) \
        "$([ $result != PASS ] && echo "status $status, $out/$name.log")")")
}
# skip NAME WHY: a step that cannot run here.
skip() {
    printf '%-24s SKIP (%s)\n' "$1" "$2"
    rows+=("$(printf '%-24s %-5s %8s  %s' "$1" SKIP "" "$2")")
    skipped=1
}

echo "== host part, tree $W, logs in $out"
step fmt "$cargo" fmt --all --check
step clippy "$cargo" clippy --workspace --all-targets -- -D warnings
step unit-tests tests/scripts/untouched-home.sh "$cargo" test --workspace
step build scripts/build.sh
if [ -f "$W/dist/lib/libcuda.so.1" ]; then
    step fit_api tests/proxy/fit_api.sh
    step marker tests/proxy/marker.sh
    step graph_fit tests/proxy/graph_fit.sh
    step lifecycle tests/proxy/lifecycle.sh
    step library_entries python3 tests/proxy/library_entries.py dist/lib/libcuda.so.1
    step readonly_tables python3 tests/proxy/readonly_tables.py dist/lib/libcuda.so.1
    step driver_capability python3 tests/proxy/driver_capability.py dist/lib/libcuda.so.1
    step link_refusals python3 tests/proxy/link_refusals.py dist/lib/libcuda.so.1
    step nccl_remedy python3 tests/proxy/nccl_remedy.py dist/lib/libcuda.so.1
    step reported_traps python3 tests/proxy/reported_traps.py dist/lib/libcuda.so.1
    step launch_guard python3 tests/proxy/launch_guard.py dist/lib/libcuda.so.1
    step fault python3 tests/proxy/fault.py dist/lib/libcuda.so.1
    step driver_calls python3 tests/proxy/driver_calls.py dist/lib/libcuda.so.1
    step guard_stress tests/proxy/guard_stress.sh
    step jitlink_fork python3 tests/proxy/jitlink_fork.py dist/lib/libnvJitLink.so.13
    step safe_mode python3 tests/proxy/safe_mode.py dist/lib/libcuda.so.1
    step log_lines python3 tests/proxy/log_lines.py dist/lib/libcuda.so.1
    step summary_file python3 tests/proxy/summary_file.py dist/lib/libcuda.so.1
    step summary_fill python3 tests/proxy/summary_fill.py dist/lib/libcuda.so.1
    step nocode_launch python3 tests/proxy/nocode_launch.py dist/lib/libcuda.so.1
    step told python3 tests/proxy/told.py dist/lib/libcuda.so.1
    step profile_failure python3 tests/proxy/profile_failure.py dist/lib/libcuda.so.1
    step ptx_version python3 tests/proxy/ptx_version.py dist/lib/libcuda.so.1
    step parallel_threads python3 tests/proxy/parallel_threads.py dist/lib/libcuda.so.1
    step nvml_capability python3 tests/proxy/nvml_capability.py dist/lib/libvf_audit.so
else
    skip proxy-suites "no dist/lib/libcuda.so.1: the build failed"
fi
step stack tests/forward/stack.sh
step kernel-log tests/scripts/kernel-log.sh
step forward-bf16-host tests/forward/bf16.sh
step launcher-dash dash tests/launcher/launcher.sh
step launcher-bash bash --posix tests/launcher/launcher.sh
step torch-workers python3 tests/gpu/torch-workers.py
step vllm-engine-verdicts python3 tests/gpu/vllm-engine-verdicts.py
step triton-tc python3 tests/python/triton_tc.py
step install-syntax sh -c 'dash -n install.sh && bash --posix -n install.sh'
step install-dash python3 tests/launcher/install.py
step install-bash env VF_TEST_SHELL=bash python3 tests/launcher/install.py
step fetch-redist python3 tests/launcher/fetch-redist.py
step target-dir tests/scripts/target-dir.sh --build
if [ -d "$W/toolkits/cuda-12.9.2" ]; then
    step jitlink_objects python3 tests/proxy/jitlink_objects.py dist/lib/libnvJitLink.so.13 toolkits/cuda-12.9.2
    step bf16-kernels scripts/build-bf16-kernels.sh --check
    step nvcc-host tests/scripts/nvcc-host.sh
else
    skip jitlink_objects "no toolkits/cuda-12.9.2 (scripts/fetch-all.sh redist)"
    skip bf16-kernels "no toolkits/cuda-12.9.2 (scripts/fetch-all.sh redist)"
    skip nvcc-host "no toolkits/cuda-12.9.2 (scripts/fetch-all.sh redist)"
fi
if [ -f "$W/dist/forward/libcublasLt.so.13" ]; then
    step package-refusals tests/forward/package-refusals.sh
else
    skip package-refusals "no forwarding libraries in dist/forward (scripts/build-forwarders.sh)"
fi
if [ -d "$W/src/denvdis/data" ]; then
    # The test kernels, and PyTorch's cubins when TORCH names them.
    [ -d "$TORCH" ] || echo "   (no cubins at TORCH=$TORCH: compare.sh takes the test kernels only)"
    step compare env TORCH="$TORCH" VF_TOOL=dist/bin/vf-translate tests/sass/compare.sh
else
    skip compare "no src/denvdis (scripts/fetch-all.sh denvdis)"
fi

if [ $full = 1 ]; then
    echo "== the validator over the corpora"
    if [ -d "$TORCH" ]; then
        step validate-corpus tests/sass/validate-corpus.sh "$out/validate-corpus" "$TORCH"
    else
        skip validate-corpus "no cubins at TORCH=$TORCH"
    fi
    if [ -n "$HELDOUT" ] && [ -d "$HELDOUT" ]; then
        step heldout-validate tests/sass/heldout-validate.sh "$HELDOUT" "$out/heldout"
    else
        skip heldout-validate "HELDOUT does not name the held-out corpus (tests/sass/heldout-extract.sh)"
    fi
    step controls env TORCH="$TORCH" HELDOUT="$HELDOUT" tests/sass/controls.sh "$out/controls"
fi

# ---- the GPU part ----------------------------------------------------------------------------------
tree=${VF_RELEASE_GPU_TREE:-$W}
# Each line: NAME|SETTINGS|SCRIPT AND ARGUMENTS ("$gpu": the GPU's PCI-order index; a script that takes
# none is given it in CUDA_VISIBLE_DEVICES).
gpu_suites=(
    "run||tests/gpu/run.sh \"\$gpu\""
    "run-8.0|VF_CAPABILITY=8.0|tests/gpu/run.sh \"\$gpu\""
    "forward||tests/gpu/forward.sh \"\$gpu\""
    "forward-bf16||tests/gpu/forward-bf16.sh \"\$gpu\""
    "torch||tests/gpu/torch.sh \"\$gpu\""
    "torch-8.0|VF_CAPABILITY=8.0|tests/gpu/torch.sh \"\$gpu\""
    "torch-8.0-all|VF_CAPABILITY=8.0 VF_CAPABILITY_ALL=1|tests/gpu/torch.sh \"\$gpu\""
    "torch-models||tests/gpu/torch-models.sh \"\$gpu\""
    "torch-models-8.0|VF_CAPABILITY=8.0|tests/gpu/torch-models.sh \"\$gpu\""
    "torch-models-8.0-all|VF_CAPABILITY=8.0 VF_CAPABILITY_ALL=1|tests/gpu/torch-models.sh \"\$gpu\""
    "triton||CUDA_VISIBLE_DEVICES=\"\$gpu\" tests/gpu/triton.sh"
    "vllm-topk-topp||CUDA_VISIBLE_DEVICES=\"\$gpu\" tests/gpu/vllm-topk-topp-run.sh"
    "llamacpp||tests/gpu/llamacpp.sh \"\$gpu\""
    "launcher||tests/gpu/launcher.sh \"\$gpu\""
    "vllm|${VF_VLLM_VENV:+VF_VLLM_VENV=$VF_VLLM_VENV}|tests/gpu/vllm.sh \"\$gpu\""
    "sglang|${VF_SGLANG_VENV:+VF_SGLANG_VENV=$VF_SGLANG_VENV}|tests/gpu/sglang.sh \"\$gpu\""
    "zml||tests/gpu/zml.sh \"\$gpu\""
)
# The suites for a machine with two GPUs, in the same form: they take every GPU that the command sees
# (CUDA_VISIBLE_DEVICES) and end with status 2 when fewer than two are visible, which is a skip.
gpu2_suites=(
    "multi-gpu||tests/gpu/multi-gpu.sh"
    "nccl||tests/gpu/nccl.sh"
    "nccl-cuda12||tests/gpu/nccl.sh cuda12"
)
# The GPU, as the command finds it: VF_GPU, else the first of CUDA_VISIBLE_DEVICES, else 0.
pick='gpu=${VF_GPU:-${CUDA_VISIBLE_DEVICES:-0}}; gpu=${gpu%%,*}'
command_of() {
    local settings=$1 script=$2
    printf "cd %q && %s && %s%s" "$tree" "$pick" "${settings:+$settings }" "$script"
}

# Every script that a step or a GPU command runs directly is executable as git has it: a checkout
# takes the mode of the index, whatever the mode of the file here (outside a git checkout, the file's
# own is looked at). tests/gpu/triton.sh was not, and its command failed wherever it ran as printed.
for suite in "${gpu_suites[@]}" "${gpu2_suites[@]}"; do
    IFS='|' read -r name settings script <<< "$suite"
    read -r -a words <<< "$script"
    script=$(script_of "${words[@]}")
    [ -z "$script" ] || invoked+=("$script")
done
executable() {  # executable SCRIPT...: each one has the mode 100755 in the index
    local s mode bad=0 checkout=0
    [ "$(git -C "$W" rev-parse --show-toplevel 2>/dev/null)" = "$(cd "$W" && pwd -P)" ] && checkout=1
    for s in $(printf '%s\n' "$@" | sort -u); do
        if [ $checkout = 1 ]; then mode=$(git -C "$W" ls-files -s -- "$s" | cut -d' ' -f1)
        elif [ -x "$W/$s" ]; then mode=100755
        else mode="not executable"; fi
        if [ "$mode" = 100755 ]; then echo "ok    $s"; else echo "FAIL  $s: ${mode:-not in the index}"; bad=1; fi
    done
    return $bad
}
step executable executable "${invoked[@]}"

cat <<EOF

== GPU part (a Tesla V100; tests/README.md says what each suite needs and how long it takes)
Each command from the tree, on the machine with the GPU (VF_GPU: the GPU's PCI-order index):
EOF
for suite in "${gpu_suites[@]}"; do
    IFS='|' read -r name settings script <<< "$suite"
    printf '  %s\n' "$(command_of "$settings" "$script")"
done
echo "On a machine with two GPUs, both visible (exit status 2 when fewer are: a skip, not a failure):"
for suite in "${gpu2_suites[@]}"; do
    IFS='|' read -r name settings script <<< "$suite"
    printf '  %s\n' "$(command_of "$settings" "$script")"
done
cat <<EOF
(vllm.sh and sglang.sh take their engine from VF_VLLM_VENV and VF_SGLANG_VENV, or from the
directories of tests/README.md, "Settings"; zml.sh from VF_ZML and VF_ZML_MODEL. Read the Xid lines
of the kernel's log before and after each: none is expected.)

== An installed archive instead of this tree
The GPU scripts take this tree's dist/: the forwarding libraries of dist/forward (run.sh's cases of a
forwarded runtime, the map that torch.sh writes), and launcher.sh runs scripts/vf, which takes dist/.
VF_PROXY alone does not test an archive. With DIR the directory of the archive and its SHA256SUMS,
PREFIX a new directory, COPY another and VERSION the release's:
  VF_INSTALL_BASE=file://DIR VF_PREFIX=PREFIX VF_BIN=PREFIX/bin VF_INSTALL_FETCH=0 sh $W/install.sh
  REL=PREFIX/volta-forever-VERSION-linux-x86_64
  mkdir -p \$REL/cuda-12.9.2 && ln -s $W/toolkits/cuda-12.9.2/lib \$REL/cuda-12.9.2/lib   # or: \$REL/bin/vf fetch
  \$REL/bin/vf version
  mkdir COPY && git -C $W archive vVERSION | tar -x -C COPY
  for d in toolkits demos src; do ln -s $W/\$d COPY/\$d; done
  mkdir COPY/dist && ln -s \$REL/lib COPY/dist/lib && ln -s \$REL/forward COPY/dist/forward
  ln -sf \$REL/bin/vf COPY/scripts/vf
then, with VF_OUT a new directory (torch.sh writes its map there from COPY/dist/forward, which is
the archive's; a map written before from this tree's would be taken instead):
  VF_PROXY=\$REL/lib COPY/tests/gpu/run.sh GPU
  VF_PROXY=\$REL/lib COPY/tests/gpu/torch.sh GPU
  COPY/tests/gpu/launcher.sh GPU
EOF

if [ -n "${VF_RELEASE_GPU_RUN:-}" ]; then
    echo
    echo "== GPU part through VF_RELEASE_GPU_RUN=$VF_RELEASE_GPU_RUN"
    read -r -a prefix <<< "$VF_RELEASE_GPU_RUN"
    for suite in "${gpu_suites[@]}"; do
        IFS='|' read -r name settings script <<< "$suite"
        step "gpu-$name" "${prefix[@]}" "$(command_of "$settings" "$script")"
    done
    for suite in "${gpu2_suites[@]}"; do
        IFS='|' read -r name settings script <<< "$suite"
        skip_on=2 step "gpu-$name" "${prefix[@]}" "$(command_of "$settings" "$script")"
    done
fi

echo
echo "== summary"
printf '%s\n' "${rows[@]}"
if [ $failed = 1 ]; then
    echo "release-check: a step failed"
    exit 1
fi
if [ $skipped = 1 ]; then
    echo "release-check: no step failed, but some could not run here"
    exit 2
fi
echo "release-check: every step passed"
