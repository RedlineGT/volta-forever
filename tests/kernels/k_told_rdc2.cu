// The other unit of k_told_rdc.cu: the function that its kernels call. Built the same way:
//   toolkits/cuda-13.5.0/bin/nvcc -O3 -rdc=true -fatbin -gencode arch=compute_75,code=sm_75 \
//       -gencode arch=compute_80,code=sm_80 -o k_told_rdc2.135.sm75-sm80.fatbin k_told_rdc2.cu
__device__ float told_outside(float x) { return 3.0f * x + 1.0f; }
