// N4/K2 dense GEMM K loop. One TU per kernel. C[M,N] = A[M,K] B[N,K]^T, fp16 inputs, fp32 accumulate, row-major LDS.
// V: 0 sync load to LDS, 1 real scalar register staging, 3 direct no overlap (1 slot), 4 direct overlap (2 slots, 2 barriers),
//    5 direct no overlap with the cheap CK-order issue.
// CH: K width per step (32 or 64). BM: rows of A per block (128 or 256), BN: rows of B per block (128 or 256). BM*BN = 32768 or 16384.
// KK, NN: shape, compile time. STG: K-start stagger, 0 off, n: blocks start at ((bx+by)%n)*STGS K steps and wrap.
// M32: use v_mfma_f32_32x32x8f16 instead of 16x16x16 (hipBLASLt's choice); only with BM=256,BN=128,CH=32/64 layouts below.
#include "k1_common.h"
#ifndef WPE
#define ATTR
#else
#define ATTR __attribute__((amdgpu_waves_per_eu(WPE)))
#endif
#ifndef STG
#define STG 0
#endif
#ifndef STGS
#define STGS 1
#endif
static constexpr int CH = K2CH, ROWB = CH*2, TA = BM*ROWB, TB = BN*ROWB, NKI = CH/32;
static constexpr int NLA = CH*BM/512, NLB = CH*BN/512;  // direct: lines per wave per tile (256 B per line)
static constexpr int RPL = 256/ROWB, LPR = 64/RPL;      // direct: rows per line, lanes per row
static constexpr int N16 = CH/16, TPR = 2*N16;           // staged: 16-byte vectors per thread per 128 rows, threads per row
static constexpr int NVA = N16*BM/128, NVB = N16*BN/128;
static constexpr int IM = BM/32, JN = BN/32;             // 16-row MFMA tiles per wave pair
static constexpr int NCH = KK/CH;
extern "C" __global__ void __launch_bounds__(256) ATTR KNAME(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {
  LDSDECL;
  char* L = (char*)buf_dyn_shmem;
  const int t = (int)threadIdx.x, l = t & 63;
  const int wm = (BN==128)? ((t & 127) >> 6) : (t >> 7), wn = (BN==128)? (t >> 7) : ((t & 127) >> 6);
  const int bya = (int)blockIdx.y, bxa = (int)blockIdx.x;
  float C_l[128];
  #pragma unroll
  for (int i = 0; i < 32; ++i) *(float4*)(C_l + i*4) = make_float4(0.f,0.f,0.f,0.f);
#if STG
  const int s0 = ((bxa + bya) % STG) * STGS;
  #define KC(c) (((c) + s0) >= NCH ? ((c) + s0 - NCH) : ((c) + s0))
#else
  #define KC(c) (c)
#endif
#if K2V >= 3
  const i32x4_t rsA = make_wave_buffer_resource(A);
  const i32x4_t rsB = make_wave_buffer_resource(B);
  const int w = __builtin_amdgcn_readfirstlane(t) >> 6;
  hz5();
  const uint vA0 = (uint)(2*(((bya*BM + (l/LPR))*KK) + (l%LPR)*2));
  const uint vB0 = (uint)(2*(((bxa*BN + (l/LPR))*KK) + (l%LPR)*2));
  const uint rowinc = (uint)(2*RPL*KK);
  auto issue = [&](int c, int s) __attribute__((always_inline)) {
    const uint kb = (uint)(2*KC(c)*CH);
    const uint m0a = (uint)(s*(TA+TB) + w*NLA*256), m0b = (uint)(s*(TA+TB) + TA + w*NLB*256);
    uint va = vA0 + kb + (uint)(w*NLA)*rowinc, vb = vB0 + kb + (uint)(w*NLB)*rowinc;
  #if K2V == 3
    #pragma unroll
    for (int n = 0; n < NLA; ++n) { ldl(m0a + n*256, va, rsA); va += rowinc; }
    #pragma unroll
    for (int n = 0; n < NLB; ++n) { ldl(m0b + n*256, vb, rsB); vb += rowinc; }
  #else
    va -= rowinc; vb -= rowinc;
    asm volatile("s_mov_b32 m0, %1\n\t.rept %4\n\tv_add_u32 %0, %2, %0\n\tbuffer_load_dword %0, %3, 0 offen lds\n\ts_add_u32 m0, 256, m0\n\t.endr" : "+v"(va) : "s"(m0a),"s"(rowinc),"s"(rsA),"n"(NLA) : "memory","scc");
    asm volatile("s_mov_b32 m0, %1\n\t.rept %4\n\tv_add_u32 %0, %2, %0\n\tbuffer_load_dword %0, %3, 0 offen lds\n\ts_add_u32 m0, 256, m0\n\t.endr" : "+v"(vb) : "s"(m0b),"s"(rowinc),"s"(rsB),"n"(NLB) : "memory","scc");
  #endif
  };
#endif
#if K2V == 0 || K2V == 1
  half_t* Ap = A + (bya*BM + t/TPR)*KK + (t%TPR)*8;
  half_t* Bp = B + (bxa*BN + t/TPR)*KK + (t%TPR)*8;
  const int vrow = (128/N16)*KK;
  char* Ld = L + (t/TPR)*ROWB + (t%TPR)*16;
  #define LV(p,v,c) (*(uint4*)((p) + (v)*vrow + KC(c)*CH))
  #define WV(off,v,val) (*(uint4*)(Ld + (off) + (v)*(128/N16)*ROWB) = (val))
#endif
  auto math = [&](int s) __attribute__((always_inline)) {
    char* Las = L + s*(TA+TB); char* Lbs = Las + TA;
    const int kq = (l >> 4) & 3;
    for (int ki = 0; ki < NKI; ++ki) {
      half_t A_local[IM*8]; half_t B_local[JN*8];
      #pragma unroll
      for (int i = 0; i < IM; ++i) *(uint4*)(A_local + i*8) = *(uint4*)(Las + (wm*(BM/2) + i*16 + (t&15))*ROWB + ki*64 + kq*16);
      #pragma unroll
      for (int j = 0; j < JN; ++j) *(uint4*)(B_local + j*8) = *(uint4*)(Lbs + (wn*(BN/2) + j*16 + (t&15))*ROWB + ki*64 + kq*16);
      for (int kp = 0; kp < 2; ++kp) for (int i = 0; i < IM; ++i) for (int j = 0; j < JN; ++j)
        *(((float32x4*)C_l) + (i*JN + j)) = __builtin_amdgcn_mfma_f32_16x16x16f16(*(((float16x4*)B_local) + (j*2 + kp)), *(((float16x4*)A_local) + (i*2 + kp)), *(((float32x4*)C_l) + (i*JN + j)), 0, 0, 0);
    }
  };
#if K2V == 0
  for (int c = 0; c < NCH; ++c) {
    __syncthreads();
    #pragma unroll
    for (int v = 0; v < NVA; ++v) WV(0, v, LV(Ap, v, c));
    #pragma unroll
    for (int v = 0; v < NVB; ++v) WV(TA, v, LV(Bp, v, c));
    __syncthreads();
    math(0);
  }
#elif K2V == 1
  uint4 a0 = LV(Ap,0,0), a1 = LV(Ap,1,0), b0 = LV(Bp,0,0);
  #if BM==256 || K2CH==64
  uint4 a2 = LV(Ap,2,0), a3 = LV(Ap,3,0);
  #endif
  #if BM==256 && K2CH==64
  uint4 a4 = LV(Ap,4,0), a5 = LV(Ap,5,0), a6 = LV(Ap,6,0), a7 = LV(Ap,7,0);
  #endif
  #if BN==256 || K2CH==64
  uint4 b1 = LV(Bp,1,0);
  #endif
  #if BN==256 || K2CH==64
  uint4 b2 = LV(Bp,2,0), b3 = LV(Bp,3,0);
  #endif
  #if BN==256 && K2CH==64
  uint4 b4 = LV(Bp,4,0), b5 = LV(Bp,5,0), b6 = LV(Bp,6,0), b7 = LV(Bp,7,0);
  #endif
  // NVA, NVB fix which of these exist. BN==128, CH==32: NVB=2, handled below.
  #if BN==128 && K2CH==32
  uint4 b1x = LV(Bp,1,0);
  #endif
  for (int c = 0; c < NCH; ++c) {
    __syncthreads();
    WV(0,0,a0); WV(0,1,a1);
    #if BM==256 || K2CH==64
    WV(0,2,a2); WV(0,3,a3);
    #endif
    #if BM==256 && K2CH==64
    WV(0,4,a4); WV(0,5,a5); WV(0,6,a6); WV(0,7,a7);
    #endif
    WV(TA,0,b0);
    #if BN==128 && K2CH==32
    WV(TA,1,b1x);
    #else
    WV(TA,1,b1); WV(TA,2,b2); WV(TA,3,b3);
    #endif
    #if BN==256 && K2CH==64
    WV(TA,4,b4); WV(TA,5,b5); WV(TA,6,b6); WV(TA,7,b7);
    #endif
    if (c + 1 < NCH) {
      a0 = LV(Ap,0,c+1); a1 = LV(Ap,1,c+1); b0 = LV(Bp,0,c+1);
      #if BM==256 || K2CH==64
      a2 = LV(Ap,2,c+1); a3 = LV(Ap,3,c+1);
      #endif
      #if BM==256 && K2CH==64
      a4 = LV(Ap,4,c+1); a5 = LV(Ap,5,c+1); a6 = LV(Ap,6,c+1); a7 = LV(Ap,7,c+1);
      #endif
      #if BN==128 && K2CH==32
      b1x = LV(Bp,1,c+1);
      #else
      b1 = LV(Bp,1,c+1); b2 = LV(Bp,2,c+1); b3 = LV(Bp,3,c+1);
      #endif
      #if BN==256 && K2CH==64
      b4 = LV(Bp,4,c+1); b5 = LV(Bp,5,c+1); b6 = LV(Bp,6,c+1); b7 = LV(Bp,7,c+1);
      #endif
    }
    __syncthreads();
    math(0);
  }
#elif K2V == 4
  issue(0, 0);
  for (int c = 0; c < NCH; ++c) {
    if (c + 1 < NCH) {
      if (c > 0) __syncthreads();
      issue(c + 1, (c + 1) & 1);
      wait_vm<(NLA+NLB)>();
    } else wait_vm0();
    __syncthreads();
    math(c & 1);
  }
  wait_vm0();
#elif K2V == 3 || K2V == 5
  for (int c = 0; c < NCH; ++c) {
    if (c > 0) __syncthreads();
    issue(c, 0); wait_vm0();
    __syncthreads();
    math(0);
  }
#endif
  #pragma unroll
  for (int i5 = 0; i5 < 32; ++i5) {
    half_t cc[4]; uint2 u;
    float4 v_ = *(float4*)(C_l + i5*4);
    *((half_t*)(&(((half2*)(&(u.x)))->x))) = (half_t)(v_.x);
    *((half_t*)(&(((half2*)(&(u.x)))->y))) = (half_t)(v_.y);
    *((half_t*)(&(((half2*)(&(u.y)))->x))) = (half_t)(v_.z);
    *((half_t*)(&(((half2*)(&(u.y)))->y))) = (half_t)(v_.w);
    *(uint2*)(cc) = u;
    *(uint2*)(C + (((bya*BM + wm*(BM/2) + (i5/JN)*16 + (t&15))*NN) + (bxa*BN + wn*(BN/2) + (i5%JN)*16 + ((t&63)>>4)*4))) = *(uint2*)(cc);
  }
}
