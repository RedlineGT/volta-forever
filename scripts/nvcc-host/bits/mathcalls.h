/* Part of volta-forever (scripts/nvcc-host.sh), not of the C library: a header in front of the C
 * library's bits/mathcalls.h, which <math.h> includes once for each floating-point type.
 *
 * glibc declares sinpi and cospi from 2.41 and rsqrt from 2.42 (ISO C23) whenever _GNU_SOURCE is
 * defined, as g++ always defines it, and declares them __THROW (noexcept in C++). The CUDA 12
 * headers (crt/math_functions.h, which nvcc includes in every CUDA source) declare the same
 * functions before that, without it, and nvcc's front end refuses the second declaration
 * ("exception specification is incompatible with that of previous function"), as g++ does. The
 * headers of CUDA 13.0 and 13.1 still declare rsqrt so. scripts/nvcc-host.sh gives this directory
 * to nvcc (-isystem) only when a compile of a probe fails without it and succeeds with it.
 *
 * Here, in CUDA C++ only, while the C library declares its double and float functions, the three
 * names are macros for names of ours: the C library declares other names, which nothing calls,
 * and CUDA's declarations and definitions of rsqrt, rsqrtf, sinpi, sinpif, cospi and cospif are the
 * only ones, as with a C library older than 2.41. The long double and _FloatN functions keep
 * their names. Where a CUDA header handles the C library itself (13.x for sinpi and cospi, 13.2
 * and later for rsqrt: the macros that it defines for that), the names are left alone. */

#if defined __cplusplus && defined __CUDACC__ && defined __MATH_PRECNAME
/* (The precision: __MATH_PRECNAME pastes nothing to a name for double, f for float, l and the
   _FloatN suffixes for the others.) */
# define __vf_mathcalls_precision_ 1
# define __vf_mathcalls_precision_f 1
# if __MATH_PRECNAME (__vf_mathcalls_precision_, )
#  if !defined __NV_GLIBC_PROVIDES_IEC_60559_FUNCS
#   define __vf_mathcalls_sinpi_cospi 1
#   pragma push_macro ("sinpi")
#   pragma push_macro ("cospi")
#   undef sinpi
#   undef cospi
#   define sinpi __vf_glibc_sinpi
#   define cospi __vf_glibc_cospi
#   define __DECL_SIMD___vf_glibc_sinpi
#   define __DECL_SIMD___vf_glibc_sinpif
#   define __DECL_SIMD___vf_glibc_cospi
#   define __DECL_SIMD___vf_glibc_cospif
#  endif
#  if !defined _NV_RSQRT_SPECIFIER
#   define __vf_mathcalls_rsqrt 1
#   pragma push_macro ("rsqrt")
#   undef rsqrt
#   define rsqrt __vf_glibc_rsqrt
#   define __DECL_SIMD___vf_glibc_rsqrt
#   define __DECL_SIMD___vf_glibc_rsqrtf
#  endif
# endif
# undef __vf_mathcalls_precision_
# undef __vf_mathcalls_precision_f
#endif

#include_next <bits/mathcalls.h>

#ifdef __vf_mathcalls_sinpi_cospi
# undef __vf_mathcalls_sinpi_cospi
# undef __DECL_SIMD___vf_glibc_sinpi
# undef __DECL_SIMD___vf_glibc_sinpif
# undef __DECL_SIMD___vf_glibc_cospi
# undef __DECL_SIMD___vf_glibc_cospif
# undef sinpi
# undef cospi
# pragma pop_macro ("sinpi")
# pragma pop_macro ("cospi")
#endif
#ifdef __vf_mathcalls_rsqrt
# undef __vf_mathcalls_rsqrt
# undef __DECL_SIMD___vf_glibc_rsqrt
# undef __DECL_SIMD___vf_glibc_rsqrtf
# undef rsqrt
# pragma pop_macro ("rsqrt")
#endif
