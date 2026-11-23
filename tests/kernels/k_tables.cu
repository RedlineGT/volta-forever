// Lookup tables of internal linkage, as programs write them: `static const __device__` (in the
// global state space), `static __constant__` (in the constant one), and one in a namespace, whose
// name nvcc mangles. `reader` reads the three; `other` reads none. tests/gpu/tables.py asks for
// them by name, through NVIDIA's driver and through the layer.
//
// k_tables.129.sm70.ptx: CUDA 12.9.2, nvcc -arch=compute_70 -ptx (compiled by NVIDIA's driver)
// k_tables.135.sm75.ptx: CUDA 13.5.0, nvcc -arch=compute_75 -ptx (adapted by the layer)
// Both name the first two `table` and `ctable` (so does the symbol table of the cubin that
// nvcc -arch=sm_70 -cubin makes, as local symbols) and the third
// _ZN40_INTERNAL_61e6d877_11_k_tables_cu_reader2ns6ntableE.
static const __device__ unsigned int table[4] = {11, 22, 33, 44};
static __constant__ unsigned int ctable[4] = {55, 66, 77, 88};
namespace ns {
static const __device__ unsigned int ntable[4] = {100, 200, 300, 400};
}

extern "C" __global__ void reader(unsigned int *out) {
    unsigned int i = threadIdx.x & 3;
    out[threadIdx.x] = table[i] + ctable[i] + ns::ntable[i];
}

extern "C" __global__ void other(unsigned int *out) { out[threadIdx.x] = 7; }
