#include <hip/hip_runtime.h>
#include <tl_templates/hip/gemm.h>
typedef float float32x16_ __attribute__((ext_vector_type(16)));
typedef float float32x4_ __attribute__((ext_vector_type(4)));
struct Tr { unsigned long long re,r0,r1,me,m0,m1; unsigned hw; unsigned bk; };
#define HWID_SIMM 63492 /* hwreg(id=4 HW_ID, offset 0, size 32) */

// verbatim copy of analysis/mb/gen.py mfma_only
extern "C" __global__ void __launch_bounds__(256) mfma_only(half_t* __restrict__ A, half_t* __restrict__ B, float* __restrict__ C, int nk, int bar) {
  int t=(int)threadIdx.x;
  float C_l[64]; half_t Al[32], Bl[32];
  #pragma unroll
  for (int i=0;i<16;++i) *(float4*)(C_l+i*4)=make_float4(0.f,0.f,0.f,0.f);
  #pragma unroll
  for (int i=0;i<4;++i){ *(uint4*)(Al+i*8)=*(uint4*)(A+(i*4096)+(t*8)); *(uint4*)(Bl+i*8)=*(uint4*)(B+(i*4096)+(t*8)); }
  for (int k=0;k<nk;++k){
    if (bar){ __syncthreads(); __syncthreads(); }
    for (int kp=0;kp<4;++kp) for (int i=0;i<4;++i) for (int j=0;j<4;++j)
      *(((float32x4*)C_l)+i*4+j)=__builtin_amdgcn_mfma_f32_16x16x16f16(*(((float16x4*)Bl)+j*2+(kp&1)),*(((float16x4*)Al)+i*2+(kp&1)),*(((float32x4*)C_l)+i*4+j),0,0,0);
  }
  float s=0.f;
  #pragma unroll
  for (int i=0;i<64;++i) s+=C_l[i];
  C[t]=s;
}
// same flops per K step per wave: 2x2x8=32 MFMA x 16384 flops = 64 x 8192
extern "C" __global__ void __launch_bounds__(256) mfma32_only(half_t* __restrict__ A, half_t* __restrict__ B, float* __restrict__ C, int nk, int bar) {
  int t=(int)threadIdx.x;
  float C_l[64]; half_t Al[32], Bl[32];
  #pragma unroll
  for (int i=0;i<16;++i) *(float4*)(C_l+i*4)=make_float4(0.f,0.f,0.f,0.f);
  #pragma unroll
  for (int i=0;i<4;++i){ *(uint4*)(Al+i*8)=*(uint4*)(A+(i*4096)+(t*8)); *(uint4*)(Bl+i*8)=*(uint4*)(B+(i*4096)+(t*8)); }
  for (int k=0;k<nk;++k){
    if (bar){ __syncthreads(); __syncthreads(); }
    for (int kp=0;kp<8;++kp) for (int i=0;i<2;++i) for (int j=0;j<2;++j)
      *(((float32x16_*)C_l)+i*2+j)=__builtin_amdgcn_mfma_f32_32x32x8f16(*(((float16x4*)Bl)+j*4+(kp&3)),*(((float16x4*)Al)+i*4+(kp&3)),*(((float32x16_*)C_l)+i*2+j),0,0,0);
  }
  float s=0.f;
  #pragma unroll
  for (int i=0;i<64;++i) s+=C_l[i];
  C[t]=s;
}
// traced versions: per wave three timestamps of both clocks + HW_ID, written by the first lane after the end stamp
#define TR_HEAD \
  int t=(int)threadIdx.x; unsigned long long re=__builtin_amdgcn_s_memrealtime(), me=__builtin_amdgcn_s_memtime(); unsigned hw=(unsigned)__builtin_amdgcn_s_getreg(HWID_SIMM); \
  float C_l[64]; half_t Al[32], Bl[32]; \
  _Pragma("unroll") for (int i=0;i<16;++i) *(float4*)(C_l+i*4)=make_float4(0.f,0.f,0.f,0.f); \
  _Pragma("unroll") for (int i=0;i<4;++i){ *(uint4*)(Al+i*8)=*(uint4*)(A+(i*4096)+(t*8)); *(uint4*)(Bl+i*8)=*(uint4*)(B+(i*4096)+(t*8)); } \
  __syncthreads(); \
  unsigned long long r0=__builtin_amdgcn_s_memrealtime(), m0=__builtin_amdgcn_s_memtime();
#define TR_TAIL \
  unsigned long long r1=__builtin_amdgcn_s_memrealtime(), m1=__builtin_amdgcn_s_memtime(); \
  float s=0.f; _Pragma("unroll") for (int i=0;i<64;++i) s+=C_l[i]; C[t]=s; \
  if ((t&63)==0){ Tr* p=tr+((int)blockIdx.y*gx+(int)blockIdx.x)*4+(t>>6); p->re=re;p->r0=r0;p->r1=r1;p->me=me;p->m0=m0;p->m1=m1;p->hw=hw;p->bk=(int)blockIdx.y*gx+(int)blockIdx.x; }
extern "C" __global__ void __launch_bounds__(256) mfma_tr(half_t* __restrict__ A, half_t* __restrict__ B, float* __restrict__ C, Tr* __restrict__ tr, int nk, int gx) {
  TR_HEAD
  for (int k=0;k<nk;++k){
    for (int kp=0;kp<4;++kp) for (int i=0;i<4;++i) for (int j=0;j<4;++j)
      *(((float32x4*)C_l)+i*4+j)=__builtin_amdgcn_mfma_f32_16x16x16f16(*(((float16x4*)Bl)+j*2+(kp&1)),*(((float16x4*)Al)+i*2+(kp&1)),*(((float32x4*)C_l)+i*4+j),0,0,0);
  }
  TR_TAIL
}
extern "C" __global__ void __launch_bounds__(256) mfma32_tr(half_t* __restrict__ A, half_t* __restrict__ B, float* __restrict__ C, Tr* __restrict__ tr, int nk, int gx) {
  TR_HEAD
  for (int k=0;k<nk;++k){
    for (int kp=0;kp<8;++kp) for (int i=0;i<2;++i) for (int j=0;j<2;++j)
      *(((float32x16_*)C_l)+i*2+j)=__builtin_amdgcn_mfma_f32_32x32x8f16(*(((float16x4*)Bl)+j*4+(kp&3)),*(((float16x4*)Al)+i*4+(kp&3)),*(((float32x16_*)C_l)+i*2+j),0,0,0);
  }
  TR_TAIL
}
extern "C" __global__ void __launch_bounds__(256) empty_k(float* __restrict__ C) { if ((int)blockIdx.x<0) C[0]=1.f; }
