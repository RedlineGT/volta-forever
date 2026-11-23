# The bf16 variant: binding an intercepted function on its first call, as vf_fwd_bind does for a
# forwarded one (r11 = intercept number on entry, the function's address in r11 on return; every
# register that carries an argument is kept).
    .hidden vf_icpt_bind
    .type vf_icpt_bind,@function
vf_icpt_bind:
    sub $200,%rsp
    mov %rdi,0(%rsp)
    mov %rsi,8(%rsp)
    mov %rdx,16(%rsp)
    mov %rcx,24(%rsp)
    mov %r8,32(%rsp)
    mov %r9,40(%rsp)
    mov %rax,48(%rsp)
    mov %r10,56(%rsp)
    movdqu %xmm0,64(%rsp)
    movdqu %xmm1,80(%rsp)
    movdqu %xmm2,96(%rsp)
    movdqu %xmm3,112(%rsp)
    movdqu %xmm4,128(%rsp)
    movdqu %xmm5,144(%rsp)
    movdqu %xmm6,160(%rsp)
    movdqu %xmm7,176(%rsp)
    mov %r11,%rdi
    call vf_icpt_resolve
    mov %rax,%r11
    mov 0(%rsp),%rdi
    mov 8(%rsp),%rsi
    mov 16(%rsp),%rdx
    mov 24(%rsp),%rcx
    mov 32(%rsp),%r8
    mov 40(%rsp),%r9
    mov 48(%rsp),%rax
    mov 56(%rsp),%r10
    movdqu 64(%rsp),%xmm0
    movdqu 80(%rsp),%xmm1
    movdqu 96(%rsp),%xmm2
    movdqu 112(%rsp),%xmm3
    movdqu 128(%rsp),%xmm4
    movdqu 144(%rsp),%xmm5
    movdqu 160(%rsp),%xmm6
    movdqu 176(%rsp),%xmm7
    add $200,%rsp
    ret
    .size vf_icpt_bind,.-vf_icpt_bind

    .hidden vf_icpt_slow
    .type vf_icpt_slow,@function
vf_icpt_slow:
    sub $8,%rsp
    call vf_icpt_bind
    add $8,%rsp
    jmp *%r11
    .size vf_icpt_slow,.-vf_icpt_slow
    .hidden vf_icpt_resolve
