/* Declarations of the CUDA driver, cuBLAS and cuBLASLt that the bf16 cuBLAS runtime uses.
 *
 * This project's own text, under NVIDIA's names, so that the runtime builds without NVIDIA's
 * headers (the CI has none). Only what the runtime uses is here. vf-fwdgen (bf16.rs) compares, when
 * it builds the variants, every prototype below with the CUDA 12.9.2 headers (and those of cuBLAS
 * and cuBLASLt also with 13.5's), every enumerator with the value of the same name there, the sizes
 * of the structures, and the two cuBLASLt attribute enumerations with 12.9's as complete lists. The
 * runtime never links against these functions: it takes their addresses with dlsym and calls them
 * through pointers of these types. */
#ifndef VF_BF16_DECLS_H
#define VF_BF16_DECLS_H

#include <stddef.h>
#include <stdint.h>

/* ---- CUDA driver (cuda.h) ---------------------------------------------------------------- */

typedef enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_NOT_SUPPORTED = 801
} CUresult;

typedef uint64_t cuuint64_t;
typedef unsigned long long CUdeviceptr_v2;
typedef CUdeviceptr_v2 CUdeviceptr;
typedef int CUdevice_v1;
typedef CUdevice_v1 CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUgraph_st *CUgraph;
typedef struct CUgraphNode_st *CUgraphNode;
typedef struct CUmemPoolHandle_st *CUmemoryPool;
typedef struct CUuserObject_st *CUuserObject;
typedef struct CUgraphEdgeData_st CUgraphEdgeData;
typedef void (*CUhostFn)(void *userData);

#define CU_STREAM_LEGACY ((CUstream)0x1)
#define CU_STREAM_PER_THREAD ((CUstream)0x2)

typedef enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
} CUdevice_attribute;

typedef enum CUfunction_attribute_enum {
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4
} CUfunction_attribute;

typedef enum CUstreamCaptureStatus_enum {
    CU_STREAM_CAPTURE_STATUS_NONE = 0,
    CU_STREAM_CAPTURE_STATUS_ACTIVE = 1,
    CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2
} CUstreamCaptureStatus;

typedef enum CUmemPool_attribute_enum {
    CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4
} CUmemPool_attribute;

typedef enum CUmemAllocationType_enum {
    CU_MEM_ALLOCATION_TYPE_PINNED = 1
} CUmemAllocationType;

typedef enum CUmemAllocationHandleType_enum {
    CU_MEM_HANDLE_TYPE_NONE = 0
} CUmemAllocationHandleType;

typedef enum CUmemLocationType_enum {
    CU_MEM_LOCATION_TYPE_DEVICE = 1
} CUmemLocationType;

typedef enum CUuserObject_flags_enum {
    CU_USER_OBJECT_NO_DESTRUCTOR_SYNC = 1
} CUuserObject_flags;

typedef enum CUuserObjectRetain_flags_enum {
    CU_GRAPH_USER_OBJECT_MOVE = 1
} CUuserObjectRetain_flags;

typedef struct CUmemLocation_st {
    CUmemLocationType type;
    int id;
} CUmemLocation_v1;
typedef CUmemLocation_v1 CUmemLocation;

typedef struct CUmemPoolProps_st {
    CUmemAllocationType allocType;
    CUmemAllocationHandleType handleTypes;
    CUmemLocation location;
    void *win32SecurityAttributes;
    size_t maxSize;
    unsigned short usage;
    unsigned char reserved[54];
} CUmemPoolProps_v1;
typedef CUmemPoolProps_v1 CUmemPoolProps;

CUresult cuCtxGetCurrent(CUcontext *pctx);
CUresult cuCtxGetId(CUcontext ctx, unsigned long long *ctxId);
CUresult cuCtxGetDevice(CUdevice *device);
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev);
CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx);
CUresult cuStreamGetCaptureInfo_v3(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out,
                                   CUgraph *graph_out, const CUgraphNode **dependencies_out,
                                   const CUgraphEdgeData **edgeData_out, size_t *numDependencies_out);
CUresult cuModuleLoadData(CUmodule *module, const void *image);
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
CUresult cuFuncLoad(CUfunction function);
CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc);
CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
CUresult cuMemPoolCreate(CUmemoryPool *pool, const CUmemPoolProps *poolProps);
CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value);
CUresult cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream);
CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream);
CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream);
CUresult cuUserObjectCreate(CUuserObject *object_out, void *ptr, CUhostFn destroy, unsigned int initialRefcount,
                            unsigned int flags);
CUresult cuUserObjectRelease(CUuserObject object, unsigned int count);
CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags);
CUresult cuGetErrorName(CUresult error, const char **pStr);

/* ---- library types (library_types.h) ----------------------------------------------------- */

typedef enum cudaDataType_t {
    CUDA_R_32F = 0,
    CUDA_R_16F = 2,
    CUDA_R_16BF = 14
} cudaDataType;
typedef enum cudaDataType_t cudaDataType_t;

/* ---- cuBLAS (cublas_api.h) ---------------------------------------------------------------- */

typedef struct CUstream_st *cudaStream_t;
typedef struct cublasContext *cublasHandle_t;

typedef enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15
} cublasStatus_t;

typedef enum {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2
} cublasOperation_t;

typedef enum {
    CUBLAS_POINTER_MODE_HOST = 0,
    CUBLAS_POINTER_MODE_DEVICE = 1
} cublasPointerMode_t;

typedef enum {
    CUBLAS_DEFAULT_MATH = 0,
    CUBLAS_TENSOR_OP_MATH = 1,
    CUBLAS_PEDANTIC_MATH = 2,
    CUBLAS_TF32_TENSOR_OP_MATH = 3,
    CUBLAS_FP32_EMULATED_BF16X9_MATH = 4,
    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16
} cublasMath_t;

typedef enum {
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_PEDANTIC = 69
} cublasComputeType_t;

typedef enum {
    CUBLAS_GEMM_DEFAULT = -1,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99
} cublasGemmAlgo_t;

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t *streamId);
cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId);
cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void *workspace, size_t workspaceSizeInBytes);
cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t *mode);
cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode);
cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                            int k, const void *alpha, const void *A, cudaDataType Atype, int lda, const void *B,
                            cudaDataType Btype, int ldb, const void *beta, void *C, cudaDataType Ctype, int ldc,
                            cublasComputeType_t computeType, cublasGemmAlgo_t algo);
cublasStatus_t cublasGemmEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m,
                               int64_t n, int64_t k, const void *alpha, const void *A, cudaDataType Atype,
                               int64_t lda, const void *B, cudaDataType Btype, int64_t ldb, const void *beta,
                               void *C, cudaDataType Ctype, int64_t ldc, cublasComputeType_t computeType,
                               cublasGemmAlgo_t algo);
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                          int m, int n, int k, const void *alpha, const void *A, cudaDataType Atype,
                                          int lda, long long int strideA, const void *B, cudaDataType Btype, int ldb,
                                          long long int strideB, const void *beta, void *C, cudaDataType Ctype,
                                          int ldc, long long int strideC, int batchCount,
                                          cublasComputeType_t computeType, cublasGemmAlgo_t algo);
cublasStatus_t cublasGemmStridedBatchedEx_64(cublasHandle_t handle, cublasOperation_t transa,
                                             cublasOperation_t transb, int64_t m, int64_t n, int64_t k,
                                             const void *alpha, const void *A, cudaDataType Atype, int64_t lda,
                                             long long int strideA, const void *B, cudaDataType Btype, int64_t ldb,
                                             long long int strideB, const void *beta, void *C, cudaDataType Ctype,
                                             int64_t ldc, long long int strideC, int64_t batchCount,
                                             cublasComputeType_t computeType, cublasGemmAlgo_t algo);

/* ---- cuBLASLt (cublasLt.h) ------------------------------------------------------------------ */

typedef struct cublasLtContext *cublasLtHandle_t;
typedef struct {
    uint64_t data[8];
} cublasLtMatrixLayoutOpaque_t;
typedef cublasLtMatrixLayoutOpaque_t *cublasLtMatrixLayout_t;
typedef struct {
    uint64_t data[8];
} cublasLtMatmulAlgo_t;
typedef struct {
    uint64_t data[32];
} cublasLtMatmulDescOpaque_t;
typedef cublasLtMatmulDescOpaque_t *cublasLtMatmulDesc_t;

typedef enum {
    CUBLASLT_POINTER_MODE_HOST = 0,
    CUBLASLT_POINTER_MODE_DEVICE = 1
} cublasLtPointerMode_t;

typedef enum {
    CUBLASLT_ORDER_COL = 0
} cublasLtOrder_t;

typedef enum {
    CUBLASLT_EPILOGUE_DEFAULT = 1,
    CUBLASLT_EPILOGUE_BIAS = 4
} cublasLtEpilogue_t;

/* Complete: every attribute of a matmul descriptor in CUDA 12.9 (checked by vf-fwdgen). */
typedef enum {
    CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = 0,
    CUBLASLT_MATMUL_DESC_SCALE_TYPE = 1,
    CUBLASLT_MATMUL_DESC_POINTER_MODE = 2,
    CUBLASLT_MATMUL_DESC_TRANSA = 3,
    CUBLASLT_MATMUL_DESC_TRANSB = 4,
    CUBLASLT_MATMUL_DESC_TRANSC = 5,
    CUBLASLT_MATMUL_DESC_FILL_MODE = 6,
    CUBLASLT_MATMUL_DESC_EPILOGUE = 7,
    CUBLASLT_MATMUL_DESC_BIAS_POINTER = 8,
    CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE = 10,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = 11,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = 12,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = 13,
    CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE = 14,
    CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET = 15,
    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = 17,
    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = 18,
    CUBLASLT_MATMUL_DESC_C_SCALE_POINTER = 19,
    CUBLASLT_MATMUL_DESC_D_SCALE_POINTER = 20,
    CUBLASLT_MATMUL_DESC_AMAX_D_POINTER = 21,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE = 22,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = 23,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER = 24,
    CUBLASLT_MATMUL_DESC_FAST_ACCUM = 25,
    CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = 26,
    CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS = 27,
    CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS = 28,
    CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER = 29,
    CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER = 30,
    CUBLASLT_MATMUL_DESC_A_SCALE_MODE = 31,
    CUBLASLT_MATMUL_DESC_B_SCALE_MODE = 32,
    CUBLASLT_MATMUL_DESC_C_SCALE_MODE = 33,
    CUBLASLT_MATMUL_DESC_D_SCALE_MODE = 34,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_MODE = 35,
    CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER = 36,
    CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE = 37
} cublasLtMatmulDescAttributes_t;

/* Complete: every attribute of a matrix layout in CUDA 12.9 (checked by vf-fwdgen). */
typedef enum {
    CUBLASLT_MATRIX_LAYOUT_TYPE = 0,
    CUBLASLT_MATRIX_LAYOUT_ORDER = 1,
    CUBLASLT_MATRIX_LAYOUT_ROWS = 2,
    CUBLASLT_MATRIX_LAYOUT_COLS = 3,
    CUBLASLT_MATRIX_LAYOUT_LD = 4,
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5,
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6,
    CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET = 7,
    CUBLASLT_MATRIX_LAYOUT_BATCH_MODE = 8
} cublasLtMatrixLayoutAttribute_t;

cublasStatus_t cublasLtCreate(cublasLtHandle_t *lightHandle);
cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle);
cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, const void *alpha,
                              const void *A, cublasLtMatrixLayout_t Adesc, const void *B,
                              cublasLtMatrixLayout_t Bdesc, const void *beta, const void *C,
                              cublasLtMatrixLayout_t Cdesc, void *D, cublasLtMatrixLayout_t Ddesc,
                              const cublasLtMatmulAlgo_t *algo, void *workspace, size_t workspaceSizeInBytes,
                              cudaStream_t stream);
cublasStatus_t cublasLtMatmulDescInit_internal(cublasLtMatmulDesc_t matmulDesc, size_t size,
                                               cublasComputeType_t computeType, cudaDataType_t scaleType);
cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc, cublasComputeType_t computeType,
                                        cudaDataType_t scaleType);
cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc);
cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr,
                                              const void *buf, size_t sizeInBytes);
cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr,
                                              void *buf, size_t sizeInBytes, size_t *sizeWritten);
cublasStatus_t cublasLtMatrixLayoutInit_internal(cublasLtMatrixLayout_t matLayout, size_t size, cudaDataType type,
                                                 uint64_t rows, uint64_t cols, int64_t ld);
cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout, cudaDataType type, uint64_t rows,
                                          uint64_t cols, int64_t ld);
cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout);
cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout,
                                                cublasLtMatrixLayoutAttribute_t attr, const void *buf,
                                                size_t sizeInBytes);
cublasStatus_t cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t matLayout,
                                                cublasLtMatrixLayoutAttribute_t attr, void *buf,
                                                size_t sizeInBytes, size_t *sizeWritten);

#endif
