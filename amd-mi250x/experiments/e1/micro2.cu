#include <hip/hip_runtime.h>
#include <tl_templates/hip/gemm.h>
struct Tr { unsigned long long re,r0,r1,me,m0,m1; unsigned hw; unsigned bk; };
// clock reference kernels: same stamps, no MFMA. nops: scalar only. valu: fp32 fused math in 64 regs per lane
extern "C" __global__ void __launch_bounds__(256) nop_tr(float* __restrict__ C, Tr* __restrict__ tr, int nk, int gx) {
  int t=(int)threadIdx.x; unsigned long long re=__builtin_amdgcn_s_memrealtime(), me=__builtin_amdgcn_s_memtime(); unsigned hw=(unsigned)__builtin_amdgcn_s_getreg(63492);
  __syncthreads();
  unsigned long long r0=__builtin_amdgcn_s_memrealtime(), m0=__builtin_amdgcn_s_memtime();
  for (int k=0;k<nk;++k){ asm volatile("s_nop 15\n s_nop 15\n s_nop 15\n s_nop 15\n s_nop 15\n s_nop 15\n s_nop 15\n s_nop 15\n"); }
  unsigned long long r1=__builtin_amdgcn_s_memrealtime(), m1=__builtin_amdgcn_s_memtime();
  C[t]=1.f;
  if ((t&63)==0){ Tr* p=tr+((int)blockIdx.y*gx+(int)blockIdx.x)*4+(t>>6); p->re=re;p->r0=r0;p->r1=r1;p->me=me;p->m0=m0;p->m1=m1;p->hw=hw;p->bk=(int)blockIdx.y*gx+(int)blockIdx.x; }
}
extern "C" __global__ void __launch_bounds__(256) valu_tr(float* __restrict__ C, Tr* __restrict__ tr, int nk, int gx) {
  int t=(int)threadIdx.x; unsigned long long re=__builtin_amdgcn_s_memrealtime(), me=__builtin_amdgcn_s_memtime(); unsigned hw=(unsigned)__builtin_amdgcn_s_getreg(63492);
  __syncthreads();
  unsigned long long r0=__builtin_amdgcn_s_memrealtime(), m0=__builtin_amdgcn_s_memtime();
  float x[16]; 
  #pragma unroll
  for (int i=0;i<16;++i) x[i]=(float)(t+i)*1e-3f;
  for (int k=0;k<nk;++k){
    #pragma unroll
    for (int i=0;i<16;++i) x[i]=x[i]*1.0001f+x[(i+1)&15]*0.5f;
    #pragma unroll
    for (int i=0;i<16;++i) x[i]=x[i]*0.4f+0.1f;
  }
  unsigned long long r1=__builtin_amdgcn_s_memrealtime(), m1=__builtin_amdgcn_s_memtime();
  float s=0; 
  #pragma unroll
  for (int i=0;i<16;++i) s+=x[i];
  C[t]=s;
  if ((t&63)==0){ Tr* p=tr+((int)blockIdx.y*gx+(int)blockIdx.x)*4+(t>>6); p->re=re;p->r0=r0;p->r1=r1;p->me=me;p->m0=m0;p->m1=m1;p->hw=hw;p->bk=(int)blockIdx.y*gx+(int)blockIdx.x; }
}
// MFMA on the first wave only: one SIMD of four busy
extern "C" __global__ void __launch_bounds__(256) mfma_w1_tr(half_t* __restrict__ A, half_t* __restrict__ B, float* __restrict__ C, Tr* __restrict__ tr, int nk, int gx) {
  int t=(int)threadIdx.x; unsigned long long re=__builtin_amdgcn_s_memrealtime(), me=__builtin_amdgcn_s_memtime(); unsigned hw=(unsigned)__builtin_amdgcn_s_getreg(63492);
  float C_l[64]; half_t Al[32], Bl[32];
  #pragma unroll
  for (int i=0;i<16;++i) *(float4*)(C_l+i*4)=make_float4(0.f,0.f,0.f,0.f);
  #pragma unroll
  for (int i=0;i<4;++i){ *(uint4*)(Al+i*8)=*(uint4*)(A+(i*4096)+(t*8)); *(uint4*)(Bl+i*8)=*(uint4*)(B+(i*4096)+(t*8)); }
  __syncthreads();
  unsigned long long r0=__builtin_amdgcn_s_memrealtime(), m0=__builtin_amdgcn_s_memtime();
  if ((t>>6)==0) for (int k=0;k<nk;++k){
    for (int kp=0;kp<4;++kp) for (int i=0;i<4;++i) for (int j=0;j<4;++j)
      *(((float32x4*)C_l)+i*4+j)=__builtin_amdgcn_mfma_f32_16x16x16f16(*(((float16x4*)Bl)+j*2+(kp&1)),*(((float16x4*)Al)+i*2+(kp&1)),*(((float32x4*)C_l)+i*4+j),0,0,0);
  }
  unsigned long long r1=__builtin_amdgcn_s_memrealtime(), m1=__builtin_amdgcn_s_memtime();
  float s=0; 
  #pragma unroll
  for (int i=0;i<64;++i) s+=C_l[i];
  C[t]=s;
  if ((t&63)==0){ Tr* p=tr+((int)blockIdx.y*gx+(int)blockIdx.x)*4+(t>>6); p->re=re;p->r0=r0;p->r1=r1;p->me=me;p->m0=m0;p->m1=m1;p->hw=hw;p->bk=(int)blockIdx.y*gx+(int)blockIdx.x; }
}
