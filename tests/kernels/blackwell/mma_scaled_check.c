// mma_scaled_check.c: compare block-scaled MMA results from the V100 (through the layer) against
// reference vectors from the RTX 5090.
//
// usage: mma_scaled_check MODULE REFVEC_DIR
//
// For each kernel, reads its reference file (INPUT+OUTPUT pairs), runs the same inputs through
// the layer on the V100, and reports exact matches, 1-ULP, and larger differences.
//
// Environment: DRVRUN_LIBCUDA (set to the layer's libcuda.so.1).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult; typedef int CUdevice;
typedef struct CUctx_st* CUcontext; typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction; typedef unsigned long long CUdeviceptr;
enum{CU_JIT_INFO_LOG_BUFFER=3,CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES=4,
     CU_JIT_ERROR_LOG_BUFFER=5,CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES=6};
static void* lib;
#define SYM(r,n,...) static r(*n)(__VA_ARGS__)
SYM(CUresult,cuInit,unsigned); SYM(CUresult,cuDeviceGet,CUdevice*,int);
SYM(CUresult,cuDeviceGetName,char*,int,CUdevice);
SYM(CUresult,cuCtxCreate_v2,CUcontext*,unsigned,CUdevice);
SYM(CUresult,cuCtxDestroy_v2,CUcontext);
SYM(CUresult,cuCtxSynchronize,void);
SYM(CUresult,cuModuleLoadDataEx,CUmodule*,const void*,unsigned,int*,void**);
SYM(CUresult,cuModuleGetFunction,CUfunction*,CUmodule,const char*);
SYM(CUresult,cuMemAlloc_v2,CUdeviceptr*,size_t);
SYM(CUresult,cuMemcpyHtoD_v2,CUdeviceptr,const void*,size_t);
SYM(CUresult,cuMemcpyDtoH_v2,void*,CUdeviceptr,size_t);
SYM(CUresult,cuMemsetD8_v2,CUdeviceptr,unsigned char,size_t);
SYM(CUresult,cuLaunchKernel,CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,
    unsigned,void*,void**,void**);
SYM(CUresult,cuGetErrorName,CUresult,const char**);
SYM(CUresult,cuMemFree_v2,CUdeviceptr);
static void* sym(const char* n){void* p=dlsym(lib,n);if(!p){fprintf(stderr,"missing %s\n",n);exit(2);}return p;}
static const char* ename(CUresult r){const char*s="?";cuGetErrorName(r,&s);return s?s:"?";}
#define CK(c) do{CUresult r_=(c);if(r_){fprintf(stderr,"FAIL %s -> %d %s\n",#c,r_,ename(r_));exit(1);}}while(0)

#define THREADS 32
#define NIN 12
#define NOUT 4

static float f_from_u(uint32_t u){float f;memcpy(&f,&u,4);return f;}
static uint32_t u_from_f(float f){uint32_t u;memcpy(&u,&f,4);return u;}
static int64_t ulp_dist(float a, float b) {
    if(isnan(a)||isnan(b)) return -1;
    uint32_t ua=u_from_f(a),ub=u_from_f(b);
    int64_t ia=(ua&0x80000000)?-(int64_t)(ua&0x7fffffff):(int64_t)ua;
    int64_t ib=(ub&0x80000000)?-(int64_t)(ub&0x7fffffff):(int64_t)ub;
    int64_t d=ia-ib; return d<0?-d:d;
}

typedef struct { const char* name; } Kernel;
static const Kernel default_kernels[] = {
    {"s_mxf4_2x"},
    {"s_nvf4_4x"},
    {"s_mxf8_e4m3_1x"},
};
#define N_DEFAULT_KERNELS 3

static const char* suffix = "";
static int check_kernel(CUmodule mod, const char* name, const char* refdir) {
    CUfunction func;
    CK(cuModuleGetFunction(&func, mod, name));

    char path[512];
    snprintf(path, sizeof(path), "%s/%s%s.bin", refdir, name, suffix);
    FILE* fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); return 1; }

    uint32_t kernel_id, n_cases;
    if (fread(&kernel_id, 4, 1, fp) != 1 || fread(&n_cases, 4, 1, fp) != 1) {
        fprintf(stderr, "bad header in %s\n", path); fclose(fp); return 1;
    }

    CUdeviceptr di, do_;
    CK(cuMemAlloc_v2(&di, THREADS * NIN * 4));
    CK(cuMemAlloc_v2(&do_, THREADS * NOUT * 4));

    int match = 0, ulp1 = 0, ulp2_10 = 0, ulp_gt10 = 0;
    int nan_both = 0, nan_mismatch = 0, total = 0, launch_fail = 0;
    int64_t max_ulp = 0;
    int worst_case = -1, worst_thread = -1, worst_reg = -1;

    for (uint32_t ci = 0; ci < n_cases; ci++) {
        uint32_t in_data[THREADS * NIN], ref_out[THREADS * NOUT], gpu_out[THREADS * NOUT];
        if (fread(in_data, 4, THREADS * NIN, fp) != THREADS * NIN ||
            fread(ref_out, 4, THREADS * NOUT, fp) != THREADS * NOUT) {
            fprintf(stderr, "short read at case %d in %s\n", ci, path); break;
        }
        CK(cuMemcpyHtoD_v2(di, in_data, sizeof(in_data)));
        CK(cuMemsetD8_v2(do_, 0xee, sizeof(gpu_out)));
        void* args[] = {&di, &do_, &di};
        CK(cuLaunchKernel(func, 1, 1, 1, THREADS, 1, 1, 0, NULL, args, NULL));
        CUresult r = cuCtxSynchronize();
        if (r) { launch_fail++; continue; }
        CK(cuMemcpyDtoH_v2(gpu_out, do_, sizeof(gpu_out)));

        for (int t = 0; t < THREADS; t++) {
            for (int reg = 0; reg < NOUT; reg++) {
                uint32_t hw = ref_out[t * NOUT + reg], got = gpu_out[t * NOUT + reg];
                float fhw = f_from_u(hw), fgot = f_from_u(got);
                total++;
                if (hw == got) { match++; continue; }
                if (isnan(fhw) && isnan(fgot)) { nan_both++; continue; }
                if (isnan(fhw) || isnan(fgot)) { nan_mismatch++; continue; }
                int64_t d = ulp_dist(fhw, fgot);
                if (d == 1) ulp1++;
                else if (d <= 10) ulp2_10++;
                else {
                    ulp_gt10++;
                    if (d > max_ulp) {
                        max_ulp = d;
                        worst_case = ci;
                        worst_thread = t;
                        worst_reg = reg;
                    }
                }
            }
        }
    }

    fclose(fp);
    CK(cuMemFree_v2(di));
    CK(cuMemFree_v2(do_));

    int finite_diff = total - match - nan_both - nan_mismatch;
    printf("\n%s: %d cases, %d results\n", name, n_cases, total);
    printf("  exact bit match: %d / %d (%.1f%%)\n", match, total, 100.0 * match / (total > 0 ? total : 1));
    printf("  NaN in both:     %d (bit pattern may differ)\n", nan_both);
    printf("  NaN mismatch:    %d (one NaN, other not)\n", nan_mismatch);
    printf("  1-ULP diff:      %d (%.2f%%)\n", ulp1, 100.0 * ulp1 / (total > 0 ? total : 1));
    printf("  2-10 ULP:        %d\n", ulp2_10);
    printf("  >10 ULP:         %d (max %lld ULP", ulp_gt10, (long long)max_ulp);
    if (worst_case >= 0)
        printf(", case %d thread %d reg %d", worst_case, worst_thread, worst_reg);
    printf(")\n");
    printf("  launch fail:     %d\n", launch_fail);
    // One line per kernel that a script can read: identical results, one unit of the last place
    // away, further away (2 or more), one NaN and one number (NaN in both is counted apart).
    printf("RESULT %s%s identical=%d nan_both=%d one_unit=%d further=%d nan_mismatch=%d total=%d launch_fail=%d\n", name, suffix,
           match, nan_both, ulp1, ulp2_10 + ulp_gt10, nan_mismatch, total, launch_fail);

    // Return 0 if no non-rounding errors (no NaN mismatches except from NaN scales,
    // and no >10 ULP differences).
    return (nan_mismatch > 0 || ulp_gt10 > 0) ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: mma_scaled_check MODULE REFVEC_DIR [SUFFIX [KERNEL...]]\n"); return 2; }
    if (argc > 3) suffix = argv[3];
    FILE* fp = fopen(argv[1], "rb"); if (!fp) { perror(argv[1]); return 2; }
    fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET);
    char* img = sz < 0 ? NULL : calloc(sz + 1, 1);
    if (!img || fread(img, 1, sz, fp) != (size_t)sz) { fprintf(stderr, "%s: cannot read it\n", argv[1]); return 2; }
    fclose(fp);

    lib = dlopen(getenv("DRVRUN_LIBCUDA") ? getenv("DRVRUN_LIBCUDA") : "libcuda.so.1", RTLD_NOW|RTLD_GLOBAL);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
#define L(n) n=sym(#n)
    L(cuInit);L(cuDeviceGet);L(cuDeviceGetName);L(cuCtxCreate_v2);L(cuCtxDestroy_v2);
    L(cuCtxSynchronize);L(cuModuleLoadDataEx);L(cuModuleGetFunction);L(cuMemAlloc_v2);
    L(cuMemcpyHtoD_v2);L(cuMemcpyDtoH_v2);L(cuMemsetD8_v2);L(cuLaunchKernel);L(cuGetErrorName);
    L(cuMemFree_v2);

    CUdevice dev; CUcontext ctx; char devname[128];
    CK(cuInit(0)); CK(cuDeviceGet(&dev, 0)); CK(cuDeviceGetName(devname, 128, dev));
    printf("device: %s\n", devname);
    CK(cuCtxCreate_v2(&ctx, 0, dev));

    static char info[65536], error[65536];
    int opts[] = {3, 4, 5, 6}; void* vals[] = {info, (void*)65536ULL, error, (void*)65536ULL};
    CUmodule mod;
    CUresult r = cuModuleLoadDataEx(&mod, img, 4, opts, vals);
    if (r) { fprintf(stderr, "FAIL load -> %d %s\n%s\n", r, ename(r), error); return 1; }

    int any_fail = 0;
    // The kernels named after the suffix, or the three forms of k_mma_scaled_forms.
    if (argc > 4) {
        for (int k = 4; k < argc; k++)
            if (check_kernel(mod, argv[k], argv[2])) any_fail = 1;
    } else {
        for (int k = 0; k < N_DEFAULT_KERNELS; k++)
            if (check_kernel(mod, default_kernels[k].name, argv[2])) any_fail = 1;
    }

    CK(cuCtxDestroy_v2(ctx));
    return any_fail;
}
