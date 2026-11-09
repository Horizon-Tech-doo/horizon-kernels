// K1 micro tests: lane mapping, bad offsets, M0 hazard, vmcnt ordering, clock.
#include "k1_common.h"
#define SENT 0xDEADBEEFu
FI uint patt(uint i){ return i*2654435761u + 0x12345u; }

// ---- mapping and bad offsets ----
// Wave 0 (or every wave if allw) issues nlines direct loads. m0val and voffsets are host-chosen. All LDS is dumped.
// meta per thread, 8 dwords: lane, exec_lo/hi before, exec_lo/hi after issue, exec_lo/hi after vmcnt(0), trapsts.
extern "C" __global__ void __launch_bounds__(256) m_bad(uint* G, uint* OUT, uint* META, uint ldsdw, uint m0val, uint rangeb, uint vstart, uint allw, uint nlines, uint lanemask, uint instoff) {
  LDSDECL; volatile uint* L=(volatile uint*)buf_dyn_shmem; const int t=(int)threadIdx.x;
  for (uint i=0;i<ldsdw/256;++i) L[i*256+t]=SENT;
  __syncthreads();
  const i32x4_t rs=make_wave_buffer_resource(G,rangeb);
  const uint w=(uint)__builtin_amdgcn_readfirstlane(t)>>6;
  hz5();
  uint lane=lane_id();
  unsigned long long e0=0,e1=0,e2=0; uint ts=0;
  asm volatile("s_mov_b64 %0, exec" : "=s"(e0));
  if (allw || w==0) {
    for (uint n=0;n<nlines;++n) {
      uint m0v=m0val+(allw? w*nlines*256:0)+n*256;
      uint v=vstart+((allw? w*nlines:0)+n)*256+(uint)(t&63)*4;
      if (lanemask==0) {
        if (instoff) asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tbuffer_load_dword %1, %2, 0 offen offset:64 lds" : : "s"(m0v),"v"(v),"s"(rs) : "memory");
        else ldl(m0v,v,rs);
      } else if ((t&63)%2==0) ldl(m0v,v,rs);
    }
  }
  asm volatile("s_mov_b64 %0, exec" : "=s"(e1));
  wait_vm0();
  asm volatile("s_mov_b64 %0, exec" : "=s"(e2));
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_TRAPSTS)" : "=s"(ts));
  asm volatile("s_mov_b64 exec, %0" : : "s"(e0));   // restore in case hardware kept lanes masked
  __syncthreads();
  uint* M=META+((uint)blockIdx.x*256+t)*8;
  M[0]=lane; M[1]=(uint)e0; M[2]=(uint)(e0>>32); M[3]=(uint)e1; M[4]=(uint)(e1>>32); M[5]=(uint)e2; M[6]=(uint)(e2>>32); M[7]=ts;
  for (uint i=0;i<ldsdw/256;++i) OUT[blockIdx.x*ldsdw+i*256+t]=L[i*256+t];
}

// ---- M0 hazard ----
// 8 lines per wave, asm blocks are opaque to the hazard recognizer, so the sequence is exactly as written.
// mode 0: s_mov m0 immediately before the load (TileLang template order)
// mode 1: s_mov m0, s_nop 0, load
// mode 2: s_mov m0, VALU, load (CK order)
// mode 3: s_add_u32 m0 immediately before the load
// mode 4: tl::cp_async_gs<4>, the real TileLang template
template<int MODE> FI void haz_seq(uint m0s,uint vs,uint stride,const i32x4_t& rs){
  uint s=m0s, v=vs;
  if constexpr (MODE==0) {
    asm volatile(".rept 8\n\ts_mov_b32 m0, %0\n\tbuffer_load_dword %1, %3, 0 offen lds\n\ts_add_u32 %0, %0, 256\n\tv_add_u32 %1, %2, %1\n\t.endr" : "+s"(s),"+v"(v) : "s"(stride),"s"(rs) : "memory","scc");
  } else if constexpr (MODE==1) {
    asm volatile(".rept 8\n\ts_mov_b32 m0, %0\n\ts_nop 0\n\tbuffer_load_dword %1, %3, 0 offen lds\n\ts_add_u32 %0, %0, 256\n\tv_add_u32 %1, %2, %1\n\t.endr" : "+s"(s),"+v"(v) : "s"(stride),"s"(rs) : "memory","scc");
  } else if constexpr (MODE==2) {
    asm volatile("v_subrev_u32 %1, %2, %1\n\t.rept 8\n\ts_mov_b32 m0, %0\n\tv_add_u32 %1, %2, %1\n\tbuffer_load_dword %1, %3, 0 offen lds\n\ts_add_u32 %0, %0, 256\n\t.endr" : "+s"(s),"+v"(v) : "s"(stride),"s"(rs) : "memory","scc");
  } else if constexpr (MODE==3) {
    asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tbuffer_load_dword %1, %3, 0 offen lds\n\tv_add_u32 %1, %2, %1\n\t.rept 7\n\ts_add_u32 m0, 256, m0\n\tbuffer_load_dword %1, %3, 0 offen lds\n\tv_add_u32 %1, %2, %1\n\t.endr" : "+s"(s),"+v"(v) : "s"(stride),"s"(rs) : "memory","scc");
  } else if constexpr (MODE==5) {
    // VALU writes the soffset SGPR, VMEM reads it with 0 wait states. M0 was written before the VALU.
    uint sg, vl=(uint)(threadIdx.x&63)*4, vu=vs-((uint)(threadIdx.x&63)*4)-stride, st0=m0s-256;
    asm volatile("s_mov_b32 m0, %3\n\t.rept 8\n\tv_add_u32 %1, %4, %1\n\ts_add_u32 m0, 256, m0\n\tv_readfirstlane_b32 %0, %1\n\tbuffer_load_dword %2, %5, %0 offen lds\n\t.endr" : "=&s"(sg),"+v"(vu) : "v"(vl),"s"(st0),"s"(stride),"s"(rs) : "memory","scc");
  } else if constexpr (MODE>=7) {
    // sweep: 1..4 wait states
    uint sg, vl=(uint)(threadIdx.x&63)*4, vu=vs-((uint)(threadIdx.x&63)*4)-stride, st0=m0s-256;
    asm volatile("s_mov_b32 m0, %3\n\t.rept 8\n\tv_add_u32 %1, %4, %1\n\ts_add_u32 m0, 256, m0\n\tv_readfirstlane_b32 %0, %1\n\ts_nop %6\n\tbuffer_load_dword %2, %5, %0 offen lds\n\t.endr" : "=&s"(sg),"+v"(vu) : "v"(vl),"s"(st0),"s"(stride),"s"(rs),"n"(MODE-7) : "memory","scc");
  } else if constexpr (MODE==6) {
    // same, with the 5 documented wait states.
    uint sg, vl=(uint)(threadIdx.x&63)*4, vu=vs-((uint)(threadIdx.x&63)*4)-stride, st0=m0s-256;
    asm volatile("s_mov_b32 m0, %3\n\t.rept 8\n\tv_add_u32 %1, %4, %1\n\ts_add_u32 m0, 256, m0\n\tv_readfirstlane_b32 %0, %1\n\ts_nop 4\n\tbuffer_load_dword %2, %5, %0 offen lds\n\t.endr" : "=&s"(sg),"+v"(vu) : "v"(vl),"s"(st0),"s"(stride),"s"(rs) : "memory","scc");
  }
}
template<int MODE> FI void haz_body(uint* G, uint* ERR, uint* SAMP, int ntrial, uint gwords){
  LDSDECL; volatile uint* L=(volatile uint*)buf_dyn_shmem; const int t=(int)threadIdx.x;
  const uint w=(uint)__builtin_amdgcn_readfirstlane(t)>>6; const uint l=t&63;
  const i32x4_t rs=make_wave_buffer_resource(G);
  hz5();
  uint err=0, stale=0, sent=0;
  for (int tr=0; tr<ntrial; ++tr) {
    #pragma unroll
    for (int n=0;n<8;++n) L[(w*8+n)*64+l]=SENT;
    __syncthreads();
    const uint base=((uint)(blockIdx.x*8191u+tr*131u)%(gwords/4096))*1024u;      // word index, 1024 aligned
    const uint firstw=base+(w*8)*64+l;
#if 1
    if constexpr (MODE!=4) {
      haz_seq<MODE>(w*8*256, firstw*4, 256, rs);
    } else {
      #pragma unroll
      for (int n=0;n<8;++n) tl::cp_async_gs<4>((void*)(buf_dyn_shmem+((w*8+n)*64+l)*4), (const void*)(G+base+(w*8+n)*64+l));
    }
#endif
    wait_vm0();
    __syncthreads();
    #pragma unroll
    for (int n=0;n<8;++n) {
      uint got=L[(w*8+n)*64+l], exp=patt(base+(w*8+n)*64+l);
      if (got!=exp) { err++; if (got==SENT) sent++; if (n>0 && got==patt(base+(w*8+n+1)*64+l)) stale++; }
    }
  }
  uint* E=ERR+((uint)blockIdx.x*256+t)*3; E[0]=err; E[1]=sent; E[2]=stale;
}
#define HZ(m) extern "C" __global__ void __launch_bounds__(256) m_haz##m(uint* G, uint* ERR, uint* SAMP, int ntrial, uint gwords){ haz_body<m>(G,ERR,SAMP,ntrial,gwords); }
HZ(0) HZ(1) HZ(2) HZ(3) HZ(4) HZ(5) HZ(6) HZ(7) HZ(8) HZ(9) HZ(10)

// ---- vmcnt: does vmcnt(8) mean the first 8 loads have landed? ----
// Each wave issues 8 cold lines (random in a large array) then 8 hot lines (same lines each trial).
// After the wait it immediately reads the cold slots back in the same wave, with no barrier.
// WAITN 8: the CK claim. WAITN 15: positive control, only one load is guaranteed. WAITN 0: control.
template<int WAITN,int HOTFIRST> FI void ord_body(uint* G, uint* OUT, int ntrial, uint gwords, uint hotbase){
  LDSDECL; volatile uint* L=(volatile uint*)buf_dyn_shmem; const int t=(int)threadIdx.x;
  const uint w=(uint)__builtin_amdgcn_readfirstlane(t)>>6; const uint l=t&63;
  const i32x4_t rs=make_wave_buffer_resource(G);
  hz5();
  uint err=0,sent=0; unsigned long long ci=0,cw=0,cz=0;
  const uint bl=__builtin_amdgcn_readfirstlane((int)blockIdx.x);
  uint rng=(bl*0x9E3779B1u)^0x5bd1e995u^(w*77u);
  const uint nlines=gwords/64;
  for (int tr=0; tr<ntrial; ++tr) {
    #pragma unroll
    for (int n=0;n<8;++n) L[(w*n)*0+(w*16+n)*64+l]=SENT;
    asm volatile("s_waitcnt lgkmcnt(0)" : : : "memory");
    rng^=rng<<13; rng^=rng>>17; rng^=rng<<5;
    uint r=rng;
    unsigned long long c0=__builtin_amdgcn_s_memtime();
    #pragma unroll
    for (int n=0;n<8;++n) {
      const uint li=(r+(uint)n*0x10001u*97u)%nlines;
      if (HOTFIRST) ldl((w*16+8+n)*256,(hotbase+(w*16+n)*64+l)*4,rs);
      else          ldl((w*16+n)*256,(li*64+l)*4,rs);
    }
    #pragma unroll
    for (int n=0;n<8;++n) {
      const uint li=(r+(uint)n*0x10001u*97u)%nlines;
      if (HOTFIRST) ldl((w*16+n)*256,(li*64+l)*4,rs);
      else          ldl((w*16+8+n)*256,(hotbase+(w*16+n)*64+l)*4,rs);
    }
    unsigned long long c1=__builtin_amdgcn_s_memtime();
    wait_vm<WAITN>();
    unsigned long long c2=__builtin_amdgcn_s_memtime();
    uint e=0,s=0;
    #pragma unroll
    for (int n=0;n<(HOTFIRST?0:8);++n) {
      const uint li=(r+(uint)n*0x10001u*97u)%nlines;
      uint got=L[(w*16+n)*64+l]; if (got!=patt(li*64+l)) { e++; if (got==SENT) s++; }
    }
    err+=e; sent+=s;
    wait_vm0();
    unsigned long long c3=__builtin_amdgcn_s_memtime();
    ci+=c1-c0; cw+=c2-c1; cz+=c3-c2;
  }
  uint* E=OUT+((uint)blockIdx.x*256+t)*5; E[0]=err; E[1]=sent; E[2]=(uint)ci; E[3]=(uint)cw; E[4]=(uint)cz;
}
extern "C" __global__ void __launch_bounds__(256) m_ord8(uint* G, uint* OUT, int ntrial, uint gwords, uint hot){ ord_body<8,0>(G,OUT,ntrial,gwords,hot); }
extern "C" __global__ void __launch_bounds__(256) m_ord15(uint* G, uint* OUT, int ntrial, uint gwords, uint hot){ ord_body<15,0>(G,OUT,ntrial,gwords,hot); }
extern "C" __global__ void __launch_bounds__(256) m_ord0(uint* G, uint* OUT, int ntrial, uint gwords, uint hot){ ord_body<0,0>(G,OUT,ntrial,gwords,hot); }
extern "C" __global__ void __launch_bounds__(256) m_ordh(uint* G, uint* OUT, int ntrial, uint gwords, uint hot){ ord_body<8,1>(G,OUT,ntrial,gwords,hot); }

// ---- vmcnt with few TLB misses: cold lines are a sequential walk, so address translation does not stall the issue ----
// No s_memtime in the loop, so the read follows the issue as closely as possible. WAITN 15 and 16 are power controls.
template<int WAITN> FI void ords_body(uint* G, uint* OUT, int ntrial, uint gwords, uint hotbase, uint seed){
  LDSDECL; volatile uint* L=(volatile uint*)buf_dyn_shmem; const int t=(int)threadIdx.x;
  const uint w=(uint)__builtin_amdgcn_readfirstlane(t)>>6; const uint l=t&63;
  const i32x4_t rs=make_wave_buffer_resource(G);
  hz5();
  uint err=0,sent=0;
  const uint bl=__builtin_amdgcn_readfirstlane((int)blockIdx.x);
  const uint nlines=gwords/64-20000;
  uint walk=((bl*0x9E3779B1u)^(w*0x85EBCA6Bu)^(seed*0xC2B2AE35u))%nlines;
  for (int tr=0; tr<ntrial; ++tr) {
    #pragma unroll
    for (int n=0;n<8;++n) L[(w*16+n)*64+l]=SENT;
    asm volatile("s_waitcnt lgkmcnt(0)" : : : "memory");
    const uint li=walk+(uint)tr*8;
    #pragma unroll
    for (int n=0;n<8;++n) ldl((w*16+n)*256,((li+n)*64+l)*4,rs);
    #pragma unroll
    for (int n=0;n<8;++n) ldl((w*16+8+n)*256,(hotbase+(w*16+n)*64+l)*4,rs);
    wait_vm<WAITN>();
    uint e=0,s=0;
    #pragma unroll
    for (int n=0;n<8;++n) { uint got=L[(w*16+n)*64+l]; if (got!=patt((li+n)*64+l)) { e++; if (got==SENT) s++; } }
    err+=e; sent+=s;
    wait_vm0();
  }
  uint* E=OUT+((uint)blockIdx.x*256+t)*5; E[0]=err; E[1]=sent;
}
extern "C" __global__ void __launch_bounds__(256) m_ords8(uint* G, uint* OUT, int ntrial, uint gwords, uint hot, uint seed){ ords_body<8>(G,OUT,ntrial,gwords,hot,seed); }
extern "C" __global__ void __launch_bounds__(256) m_ords15(uint* G, uint* OUT, int ntrial, uint gwords, uint hot, uint seed){ ords_body<15>(G,OUT,ntrial,gwords,hot,seed); }
extern "C" __global__ void __launch_bounds__(256) m_ords16(uint* G, uint* OUT, int ntrial, uint gwords, uint hot, uint seed){ ords_body<16>(G,OUT,ntrial,gwords,hot,seed); }
extern "C" __global__ void __launch_bounds__(256) m_ords0(uint* G, uint* OUT, int ntrial, uint gwords, uint hot, uint seed){ ords_body<0>(G,OUT,ntrial,gwords,hot,seed); }

// ---- vmcnt, final form: no volatile (the memory legalizer puts vmcnt(0) around volatile LDS accesses, which voided
// m_ord* and m_ords* above: see asm of m_ords16). Reads are plain; asm memory clobbers keep them in place.
// HOTB 0: chunk 1 cold, chunk 2 cold. HOTB 1: chunk 1 cold, chunk 2 hot (an early hot return would expose reorder).
// Per-line miss counts for both chunks go to OUT.
template<int WAITN,int HOTB> FI void ordf_body(uint* G, uint* OUT, int ntrial, uint gwords, uint hotbase, uint seed){
  LDSDECL; uint* L=(uint*)buf_dyn_shmem; const int t=(int)threadIdx.x;
  const uint w=(uint)__builtin_amdgcn_readfirstlane(t)>>6; const uint l=t&63;
  const i32x4_t rs=make_wave_buffer_resource(G);
  hz5();
  uint e1[8],e2[8];
  #pragma unroll
  for (int n=0;n<8;++n){e1[n]=0;e2[n]=0;}
  const uint bl=__builtin_amdgcn_readfirstlane((int)blockIdx.x);
  const uint nlines=gwords/64-40000;
  uint walk=((bl*0x9E3779B1u)^(w*0x85EBCA6Bu)^(seed*0xC2B2AE35u))%nlines;
  for (int tr=0; tr<ntrial; ++tr) {
    #pragma unroll
    for (int n=0;n<16;++n) L[(w*16+n)*64+l]=SENT;
    asm volatile("s_waitcnt lgkmcnt(0)" : : : "memory");
    const uint li=walk+(uint)tr*16;
    #pragma unroll
    for (int n=0;n<8;++n) ldl((w*16+n)*256,((li+n)*64+l)*4,rs);
    #pragma unroll
    for (int n=0;n<8;++n) ldl((w*16+8+n)*256,((HOTB? hotbase+(w*16+n) : li+8+n)*64+l)*4,rs);
    wait_vm<WAITN>();
    #pragma unroll
    for (int n=0;n<8;++n) e1[n]+= (L[(w*16+n)*64+l]!=patt((li+n)*64+l));
    #pragma unroll
    for (int n=0;n<8;++n) e2[n]+= (L[(w*16+8+n)*64+l]!=patt((HOTB? hotbase+(w*16+n) : li+8+n)*64+l));
    wait_vm0();
  }
  uint* E=OUT+((uint)blockIdx.x*256+t)*16;
  #pragma unroll
  for (int n=0;n<8;++n){E[n]=e1[n];E[8+n]=e2[n];}
}
#define OF(w,h) extern "C" __global__ void __launch_bounds__(256) m_ordf##w##_##h(uint* G, uint* OUT, int ntrial, uint gwords, uint hot, uint seed){ ordf_body<w,h>(G,OUT,ntrial,gwords,hot,seed); }
OF(16,0) OF(15,0) OF(12,0) OF(8,0) OF(4,0) OF(0,0) OF(16,1) OF(8,1) OF(0,1)

// ---- instruction-level cost, hot and cold, cycles per wave ----
// dir: 8 buffer_load_dword lds, then vmcnt(0). x4: 2 buffer_load_dwordx4 + vmcnt(0) in one asm block, then 2 ds_write_b128.
extern "C" __global__ void __launch_bounds__(256) m_lat(uint* G, uint* OUT, int ntrial, uint gwords, uint cold){
  LDSDECL; char* L=(char*)buf_dyn_shmem; const int t=(int)threadIdx.x;
  const uint w=(uint)__builtin_amdgcn_readfirstlane(t)>>6; const uint l=t&63;
  const i32x4_t rs=make_wave_buffer_resource(G);
  hz5();
  unsigned long long cdi=0,cdw=0,cxi=0,cxw=0;
  uint rng=((uint)__builtin_amdgcn_readfirstlane((int)blockIdx.x)*0x9E3779B1u)^(w*77u)^0x1234567u;
  const uint nl=gwords/64/8;
  for (int tr=0; tr<ntrial; ++tr) {
    rng^=rng<<13; rng^=rng>>17; rng^=rng<<5;
    const uint li= cold? rng%nl : w;
    uint vb=(li*8*64+l)*4;
    unsigned long long c0=__builtin_amdgcn_s_memtime();
    #pragma unroll
    for (int n=0;n<8;++n) ldl((w*8+n)*256,vb+n*256,rs);
    unsigned long long c1=__builtin_amdgcn_s_memtime();
    wait_vm0();
    unsigned long long c2=__builtin_amdgcn_s_memtime();
    rng^=rng<<13; rng^=rng>>17; rng^=rng<<5;
    const uint li2= cold? rng%nl : w;
    uint vb2=(li2*8*64+l*8)*4;   // 32 B per lane, same 2048 B per wave as 8 direct lines
    i32x4_t a,b;
    unsigned long long c3=__builtin_amdgcn_s_memtime();
    asm volatile("buffer_load_dwordx4 %0, %2, %3, 0 offen\n\tbuffer_load_dwordx4 %1, %2, %3, 0 offen offset:16\n\ts_waitcnt vmcnt(0)" : "=&v"(a),"=&v"(b) : "v"(vb2),"s"(rs) : "memory");
    unsigned long long c4=__builtin_amdgcn_s_memtime();
    *(i32x4_t*)(L+t*16)=a; *(i32x4_t*)(L+t*16+4096)=b;
    asm volatile("s_waitcnt lgkmcnt(0)" : : : "memory");
    unsigned long long c5=__builtin_amdgcn_s_memtime();
    cdi+=c1-c0; cdw+=c2-c1; cxi+=c4-c3; cxw+=c5-c4;
  }
  uint* E=OUT+((uint)blockIdx.x*256+t)*4; E[0]=(uint)cdi; E[1]=(uint)cdw; E[2]=(uint)cxi; E[3]=(uint)cxw;
}

// ---- clock under load ----
extern "C" __global__ void __launch_bounds__(256) m_clk(half_t* A, half_t* B, uint* OUT, int nk){
  const int t=(int)threadIdx.x; float C_l[64]; half_t Al[32],Bl[32];
  #pragma unroll
  for (int i=0;i<16;++i) *(float4*)(C_l+i*4)=make_float4(0.f,0.f,0.f,0.f);
  #pragma unroll
  for (int i=0;i<4;++i){ *(uint4*)(Al+i*8)=*(uint4*)(A+(i*4096)+(t*8)); *(uint4*)(Bl+i*8)=*(uint4*)(B+(i*4096)+(t*8)); }
  unsigned long long a=__builtin_amdgcn_s_memtime(), ar=__builtin_amdgcn_s_memrealtime();
  for (int k=0;k<nk;++k) for (int kp=0;kp<4;++kp) for (int i=0;i<4;++i) for (int j=0;j<4;++j)
    *(((float32x4*)C_l)+i*4+j)=__builtin_amdgcn_mfma_f32_16x16x16f16(*(((float16x4*)Bl)+j*2+(kp&1)),*(((float16x4*)Al)+i*2+(kp&1)),*(((float32x4*)C_l)+i*4+j),0,0,0);
  unsigned long long b=__builtin_amdgcn_s_memtime(), br=__builtin_amdgcn_s_memrealtime();
  float s=0.f;
  #pragma unroll
  for (int i=0;i<64;++i) s+=C_l[i];
  uint* E=OUT+((uint)blockIdx.x*256+t)*3; E[0]=(uint)(b-a); E[1]=(uint)(br-ar); E[2]=__builtin_bit_cast(uint,s);
}
