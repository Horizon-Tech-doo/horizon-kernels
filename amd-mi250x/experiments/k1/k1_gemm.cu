// K1 dense GEMM K loop. One translation unit per kernel. C[M,N] = A[M,K] B[N,K]^T, fp16 inputs, fp32 accumulate.
// K1V: 0 base (sync load to LDS each step), 1 real register staging,
//      2 direct, overlap, 2 LDS slots, 2 barriers, safe form with s_nop between m0 and the load
//      3 direct, no overlap, 1 slot, 2 barriers, safe form
//      4 as 2 and 5 as 3 with the cheapest issue sequence (CK order: s_add m0, v_add, load)
//      6 direct, overlap, 3 LDS slots, 1 barrier: the CK ring (no barrier needed before the issue)
// Run job23501734 used an earlier V2/V3 without the barrier before the issue. That raced: a wave could overwrite
// the slot another wave was still reading. V3 at CH=64 produced wrong checksums, which is how it was found.
// CH: K width per step (32 or 64). LDS is row major with no swizzle: row stride CH*2 B.
// MATH=0: load path only. LOAD=0: first chunk only, then MFMAs on it (register/LDS math only).
#include "k1_common.h"
static constexpr int CH = K1CH;
static constexpr bool LOAD = K1LOAD;
#ifndef WPE
#define ATTR
#else
#define ATTR __attribute__((amdgpu_waves_per_eu(WPE)))
#endif
static constexpr int ROWB = CH*2;                // LDS bytes per row
static constexpr int TILE = 128*ROWB;            // bytes per tile
static constexpr int NKI  = CH/32;
static constexpr int NL   = CH/4;                // direct: lines per wave per tile (256 B per line)
static constexpr int RPL  = 256/ROWB;            // direct: rows per line
static constexpr int LPR  = 64/RPL;              // direct: lanes per row
static constexpr int N16  = CH/16;               // base, rr: 16-byte vectors per thread per tile
static constexpr int TPR  = 2*N16;               // base, rr: threads per row
extern "C" __global__ void __launch_bounds__(256) ATTR KNAME(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C, int K, int N, int nch, int bmul, int kmul) {
  LDSDECL;
  char* L = (char*)buf_dyn_shmem;
  const int t = (int)threadIdx.x, l = t & 63, wm = (t & 127) >> 6, wn = t >> 7;
  const int by = (int)blockIdx.y * bmul, bx = (int)blockIdx.x * bmul;
  float C_l[64];
#if K1MATH
  #pragma unroll
  for (int i = 0; i < 16; ++i) *(float4*)(C_l + i*4) = make_float4(0.f,0.f,0.f,0.f);
#endif
#if K1V >= 2
  const i32x4_t rsA = make_wave_buffer_resource(A);
  const i32x4_t rsB = make_wave_buffer_resource(B);
  const int w = __builtin_amdgcn_readfirstlane(t) >> 6;
  hz5();
  const uint vA0 = (uint)(2*(((by*128 + (l/LPR))*K) + (l%LPR)*2));
  const uint vB0 = (uint)(2*(((bx*128 + (l/LPR))*K) + (l%LPR)*2));
  const uint rowinc = (uint)(2*RPL*K);
  auto issue = [&](int c, int s) __attribute__((always_inline)) {
    const uint kb = (uint)(2*c*CH*kmul);
    const uint m0a = (uint)(s*2*TILE + w*NL*256), m0b = m0a + TILE;
    const uint first = (uint)(w*NL)*rowinc;
    uint va = vA0 + kb + first, vb = vB0 + kb + first;
  #if K1V == 2 || K1V == 3
    #pragma unroll
    for (int n = 0; n < NL; ++n) { ldl(m0a + n*256, va, rsA); va += rowinc; }
    #pragma unroll
    for (int n = 0; n < NL; ++n) { ldl(m0b + n*256, vb, rsB); vb += rowinc; }
  #else
    va -= rowinc; vb -= rowinc;
    asm volatile("s_mov_b32 m0, %1\n\t.rept %4\n\tv_add_u32 %0, %2, %0\n\tbuffer_load_dword %0, %3, 0 offen lds\n\ts_add_u32 m0, 256, m0\n\t.endr" : "+v"(va) : "s"(m0a),"s"(rowinc),"s"(rsA),"n"(NL) : "memory","scc");
    asm volatile("s_mov_b32 m0, %1\n\t.rept %4\n\tv_add_u32 %0, %2, %0\n\tbuffer_load_dword %0, %3, 0 offen lds\n\ts_add_u32 m0, 256, m0\n\t.endr" : "+v"(vb) : "s"(m0b),"s"(rowinc),"s"(rsB),"n"(NL) : "memory","scc");
  #endif
  };
#endif
#if K1V == 0 || K1V == 1
  half_t* Ap = A + (by*128 + ((t/TPR)))*K + (t%TPR)*8;
  half_t* Bp = B + (bx*128 + ((t/TPR)))*K + (t%TPR)*8;
  const int vrow = (128/N16)*K;
  char* Ld = L + (t/TPR)*ROWB + (t%TPR)*16;
  #define LV(p,v,c) (*(uint4*)((p) + (v)*vrow + (c)*CH*kmul))
  #define WV(off,v,val) (*(uint4*)(Ld + (off) + (v)*(128/N16)*ROWB) = (val))
#endif
#if K1MATH
  auto math = [&](int s) __attribute__((always_inline)) {
    char* Las = L + s*2*TILE; char* Lbs = Las + TILE;
    const int kq = (l >> 4) & 3;
    for (int ki = 0; ki < NKI; ++ki) {
      half_t A_local[32]; half_t B_local[32];
      #pragma unroll
      for (int i = 0; i < 4; ++i) *(uint4*)(A_local + i*8) = *(uint4*)(Las + (wm*64 + i*16 + (t&15))*ROWB + ki*64 + kq*16);
      #pragma unroll
      for (int j = 0; j < 4; ++j) *(uint4*)(B_local + j*8) = *(uint4*)(Lbs + (wn*64 + j*16 + (t&15))*ROWB + ki*64 + kq*16);
      for (int kp = 0; kp < 2; ++kp) for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
        *(((float32x4*)C_l) + (i*4 + j)) = __builtin_amdgcn_mfma_f32_16x16x16f16(*(((float16x4*)B_local) + (j*2 + kp)), *(((float16x4*)A_local) + (i*2 + kp)), *(((float32x4*)C_l) + (i*4 + j)), 0, 0, 0);
    }
  };
#else
  auto math = [&](int s) __attribute__((always_inline)) { };
#endif

#if K1V == 0
  for (int c = 0; c < nch; ++c) {
    __syncthreads();
    if (c == 0 || LOAD) {
      #pragma unroll
      for (int v = 0; v < N16; ++v) WV(0, v, LV(Ap, v, c));
      #pragma unroll
      for (int v = 0; v < N16; ++v) WV(TILE, v, LV(Bp, v, c));
    }
    __syncthreads();
    math(0);
  }
#elif K1V == 1
  uint4 a0 = LV(Ap,0,0), a1 = LV(Ap,1,0), b0 = LV(Bp,0,0), b1 = LV(Bp,1,0);
  #if K1CH == 64
  uint4 a2 = LV(Ap,2,0), a3 = LV(Ap,3,0), b2 = LV(Bp,2,0), b3 = LV(Bp,3,0);
  #endif
  for (int c = 0; c < nch; ++c) {
    __syncthreads();
    if (c == 0 || LOAD) {
      WV(0,0,a0); WV(0,1,a1); WV(TILE,0,b0); WV(TILE,1,b1);
      #if K1CH == 64
      WV(0,2,a2); WV(0,3,a3); WV(TILE,2,b2); WV(TILE,3,b3);
      #endif
      if (c + 1 < nch) {
        a0 = LV(Ap,0,c+1); a1 = LV(Ap,1,c+1); b0 = LV(Bp,0,c+1); b1 = LV(Bp,1,c+1);
        #if K1CH == 64
        a2 = LV(Ap,2,c+1); a3 = LV(Ap,3,c+1); b2 = LV(Bp,2,c+1); b3 = LV(Bp,3,c+1);
        #endif
      }
    }
    __syncthreads();
    math(0);
  }
#elif K1V == 2 || K1V == 4
  issue(0, 0);
  for (int c = 0; c < nch; ++c) {
    if (LOAD && c + 1 < nch) {
      if (c > 0) __syncthreads();            // nobody may still read the slot about to be overwritten
      issue(c + 1, (c + 1) & 1);
      wait_vm<2*NL>();
    } else wait_vm0();
    __syncthreads();
    math(LOAD ? (c & 1) : 0);
  }
  wait_vm0();
#elif K1V == 3 || K1V == 5
  for (int c = 0; c < nch; ++c) {
    if (LOAD || c == 0) {
      if (c > 0) __syncthreads();
      issue(c, 0); wait_vm0();
    }
    __syncthreads();
    math(0);
  }
#elif K1V == 6
  issue(0, 0);
  for (int c = 0; c < nch; ++c) {
    if (LOAD && c + 1 < nch) { issue(c + 1, (c + 1) % 3); wait_vm<2*NL>(); }
    else wait_vm0();
    __syncthreads();
    math(LOAD ? (c % 3) : 0);
  }
  wait_vm0();
#endif

#if K1MATH
  #pragma unroll
  for (int i5 = 0; i5 < 16; ++i5) {
    half_t cc[4]; uint2 u;
    float4 v_ = *(float4*)(C_l + i5*4);
    *((half_t*)(&(((half2*)(&(u.x)))->x))) = (half_t)(v_.x);
    *((half_t*)(&(((half2*)(&(u.x)))->y))) = (half_t)(v_.y);
    *((half_t*)(&(((half2*)(&(u.y)))->x))) = (half_t)(v_.z);
    *((half_t*)(&(((half2*)(&(u.y)))->y))) = (half_t)(v_.w);
    *(uint2*)(cc) = u;
    *(uint2*)(C + ((((int)blockIdx.y*128 + wm*64 + (i5>>2)*16 + (t&15))*N) + ((int)blockIdx.x*128 + wn*64 + (i5&3)*16 + ((t&63)>>4)*4))) = *(uint2*)(cc);
  }
#else
  *(volatile uint*)((float*)C + t) = *(volatile uint*)(L + t*4);
#endif
}
