/* Shared by the stand-ins of the host tests of the bf16 runtime (tests/forward/bf16.sh) and the test
 * program: the project's own declarations of the CUDA driver, cuBLAS and cuBLASLt (the runtime's), the
 * few more functions the stand-ins define, and the stand-ins' own controls. */
#ifndef VF_BF16_STUB_H
#define VF_BF16_STUB_H
#include "../../../crates/vf-fwdgen/bf16tc/decls.h"

#define STUB_ARGS 256 /* bytes of a kernel's argument structure that a launch event shows */

cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode);
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);

/* libcuda.so.1 stand-in */
void stub_event(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
char *stub_take_events(void);
void stub_fail(const char *name, int count);
int stub_failing(const char *name);
void stub_set_context(int index);
void stub_set_device(int device, int major, int minor);
int stub_module_loads(void);
void stub_set_stream(CUstream s, int ctx_index, unsigned long long capture, CUgraph graph);
/* libcublas.so.12 stand-in: right after the next pointer-mode query of handle h, another "thread" sets
 * its stream to s and its pointer mode to device. */
void stub_chaos(cublasHandle_t h, cudaStream_t s);
/* libcublasLt.so.12 stand-in: descriptor and layout attribute reads so far. */
int stub_lt_gets(void);
#endif
