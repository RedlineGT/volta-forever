# Sourced by tests/gpu/vllm.sh and tests/gpu/sglang.sh: the verdict of a stage, from what its
# program printed and how it ended. (Checked without a GPU by tests/gpu/vllm-engine-verdicts.py.)

refused() {
    # What the layer says when it does not give a kernel to the program.
    echo "$1" | grep -m1 'cannot run on this GPU' | cut -c1-300
}

not_given() {
    # What the layer says when the program asks an image for a kernel that the code the layer gave
    # for it has not, and the image's code for the capability told has (LIMITS.md, "The capability
    # told with VF_CAPABILITY").
    echo "$1" | grep -m1 'is not in the code that the layer gives for its image' | cut -c1-300
}

verdict_or_limit() {
    # verdict_or_limit STAGE OUTPUT WORD STATUS: as `verdict`, except for a program that ended by
    # itself with an error after the layer said that it asked for a kernel which only the code for
    # the capability told has, which the layer did not give (`not_given`), and refused nothing:
    # LIMIT, a limit that LIMITS.md documents, said with the layer's line; `failed` is not changed.
    # The same end without that line is a FAIL.
    local status=$4
    if [ "$status" != 0 ] && [ "$status" != 124 ] && [ "$status" != 137 ] && [ -n "$(not_given "$2")" ] &&
        [ -z "$(refused "$2")" ]; then
        echo "LIMIT stage-$1: $(not_given "$2")"
    else
        verdict "$@"
    fi
}

verdict() {
    # verdict STAGE OUTPUT WORD STATUS: PASS when the program's verdict is WORD and the program
    # then ended by itself with status 0 (STATUS, its exit status; 124 when it ran out of time,
    # 137 when it was killed: a program that prints its verdict and hangs as it ends does not
    # pass), else REFUSED or FAIL, and failed=1. The verdict is what follows `verdict: ` on the
    # last line of the output that starts so: what else the program, its libraries and its
    # kernels printed is not looked at for it. With ALL_MATCH for WORD, a text that leaves the
    # CPU's only where the CPU itself had a near tie (greedy_compare.py) passes too, and says so.
    # (Until 2026-11-23 the status was not looked at when the verdict was WORD.)
    local said status=$4
    said=$(echo "$2" | sed -n 's/^verdict: //p' | tail -1)
    if [ "$status" = 0 ] && [ "$said" = "$3" ]; then
        echo "PASS stage-$1"
    elif [ "$status" = 0 ] && [ "$3" = ALL_MATCH ] && [[ "$said" =~ ^MATCH_BUT_NEAR_TIES\ [0-9]+$ ]]; then
        echo "PASS stage-$1 (with ${said#MATCH_BUT_NEAR_TIES } near tie(s): see the lines above)"
    elif [ -n "$(refused "$2")" ]; then
        echo "REFUSED stage-$1: $(refused "$2")"; failed=1
    elif [ "$status" = 124 ] || [ "$status" = 137 ]; then
        echo "FAIL stage-$1: ran out of time (\$VF_ENGINE_WAIT is ${engine_wait:-?} s)${said:+, after the verdict $said}"; failed=1
    elif [ "$status" != 0 ]; then
        echo "FAIL stage-$1: the program ended with status $status${said:+, after the verdict $said}"
        echo "$2" | tail -30; failed=1
    else
        echo "FAIL stage-$1"; echo "$2" | tail -30; failed=1
    fi
}
