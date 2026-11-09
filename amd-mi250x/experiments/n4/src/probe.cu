// One-wave clock probe. Runs on a second stream while a big kernel runs. s_memtime follows the SIMD clock (E1), s_memrealtime is 25 MHz.
#include <hip/hip_runtime.h>
extern "C" __global__ void __launch_bounds__(1) clkprobe(unsigned long long* out, int idx, int spin) {
  unsigned long long r0=__builtin_amdgcn_s_memrealtime(), m0=__builtin_amdgcn_s_memtime(), r1=r0;
  while((long long)(r1-r0)<(long long)spin) r1=__builtin_amdgcn_s_memrealtime();
  unsigned long long m1=__builtin_amdgcn_s_memtime();
  out[idx*4+0]=r0; out[idx*4+1]=m0; out[idx*4+2]=r1; out[idx*4+3]=m1;
}
