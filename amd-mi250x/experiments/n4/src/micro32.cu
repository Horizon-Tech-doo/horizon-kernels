// Decode v_mfma_f32_32x32x8f16 lane layout. One wave. a,ha: lane and half that hold 1.0 in A (src1); B (src0) is lane bl, half bh, or all ones if bl<0.
#include <hip/hip_runtime.h>
#include <tl_templates/hip/gemm.h>
typedef float f32x16_ __attribute__((ext_vector_type(16)));
extern "C" __global__ void __launch_bounds__(64) m32(uint* out, int al, int ah, int bl, int bh) {
  const int l=(int)threadIdx.x&63;
  float16x4 sa={0,0,0,0}, sb={0,0,0,0};
  if(l==al) sa[ah]=1.0f16;
  if(bl<0) { sb[0]=1.0f16; sb[1]=1.0f16; sb[2]=1.0f16; sb[3]=1.0f16; } else if(l==bl) sb[bh]=1.0f16;
  f32x16_ c={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  c=__builtin_amdgcn_mfma_f32_32x32x8f16(sb,sa,c,0,0,0);
  for(int v=0;v<16;++v) { unsigned* p=(unsigned*)&out[l*16+v]; float f=c[v]; *p=(f!=0.f)?1u:0u; }
}
