/* The kernels' fat binary (kernels.fatbin, sm_70 machine code), read-only in the runtime. */
    .section .rodata.vf_bf16_fatbin,"a",@progbits
    .balign 64
    .globl vf_bf16_fatbin
    .hidden vf_bf16_fatbin
    .type vf_bf16_fatbin, @object
vf_bf16_fatbin:
    .incbin "kernels.fatbin"
    .size vf_bf16_fatbin, . - vf_bf16_fatbin
    .section .note.GNU-stack,"",@progbits
