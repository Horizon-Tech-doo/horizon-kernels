import re,os,glob,sys
R="/scratch/project_XXXXXXXXX/tilelang/cache/0.1.15/linux-x86_64/kernels/"
def src(h): return open(glob.glob(R+h+"*")[0]+"/device_kernel.cu").read()
def body(s): return s[s.index('extern "C"'):]
def rename(s,name,attr=""):
    return re.sub(r'__launch_bounds__\(256\)\s+\w+\(', '__launch_bounds__(256) '+(attr+' ' if attr else '')+name+'(', s)
def rr(s):
    """real register staging: expand v loops, a_r/b_r become uint4 scalars"""
    m=re.search(r'half_t a_r\[(\d+)\];\s*\n\s*half_t b_r\[(\d+)\];',s)
    na,nb=int(m.group(1))//8,int(m.group(2))//8
    decl=" ".join("uint4 a_r_%d;"%i for i in range(na))+" "+" ".join("uint4 b_r_%d;"%i for i in range(nb))
    s=s.replace(m.group(0),decl)
    pat=re.compile(r'for \(int (v(?:_\d+)?) = 0; \1 < (\d+); \+\+\1\) \{\n(.*?)\n\s*\}',re.S)
    def exp(mm):
        v,n,b=mm.group(1),int(mm.group(2)),mm.group(3)
        out=[]
        for i in range(n):
            t=re.sub(r'\*\(uint4\*\)\(([ab])_r \+ \(%s \* 8\)\)'%v,lambda q:"%s_r_%d"%(q.group(1),i),b)
            t=re.sub(r'\b%s\b'%v,"(%d)"%i,t)
            out.append(t)
        return "\n".join(out)
    s=pat.sub(exp,s)
    assert "a_r +" not in s and "b_r +" not in s
    return s
def strip_guard(s):
    return re.sub(r'if \(\(\(0 <= Bexp\[\(\(int\)blockIdx\.x\)\]\) && \(Bexp\[\(\(int\)blockIdx\.x\)\] < 256\)\)\) \{\s*(condval\w* = [^\n]*;)\s*\} else \{[^}]*\}',r'\1',s)
def no_epi(s):
    # replace the output epilogue by one fp32 store per thread of a sum that keeps the accumulators alive
    i=[m.start() for m in re.finditer(r'\n  if \(0 <= Bexp\[\(\(int\)blockIdx\.x\)\]\) \{\n    #pragma unroll',s)][-1]
    if 'float ac[' in s:
        keep="float s_=0.f; for(int i_=0;i_<32;++i_) s_+=ac[i_]; *(float*)(Y+((int)threadIdx.x)*2)=s_;"
    else:
        keep="float s_=0.f; for(int i_=0;i_<16;++i_) s_+=ag[i_]+au[i_]; *(float*)(Inter+((int)threadIdx.x)*2)=s_;"
    return s[:i]+"\n  "+keep+"\n  return;"+s[i:]
W=lambda n:"__attribute__((amdgpu_waves_per_eu(%d)))"%n
K=[]
add=lambda name,s:K.append(s if False else (name,s))
H={'b12832':'49a84b7117c4','b12864':'df817aeb23d4','b25632':'28b00176afce','r12832':'e4d16c90dc8c','r12864':'d0a6d9c6d104','r25632':'49685536bd2c',
   'b12864_8k':'07c5beaa2519','b25632_8k':'c5ceacd5e92d','r12864_8k':'460994032ab3','r25632_8k':'aea8968bb747',
   'b12864_32k':'52676ec39801','b25632_32k':'2fab2a272ffa'}
for n in ['b12832','b12864','b25632','b12864_8k','b25632_8k','b12864_32k','b25632_32k','r12832','r12864','r25632']:
    add(n,rename(body(src(H[n])),n))
add('b12832w4',rename(body(src(H['b12832'])),'b12832w4',W(4)))
add('b12864w4',rename(body(src(H['b12864'])),'b12864w4',W(4)))
add('b25632w2',rename(body(src(H['b25632'])),'b25632w2',W(2)))
add('b25632w2_8k',rename(body(src(H['b25632_8k'])),'b25632w2_8k',W(2)))
add('b25632w2_32k',rename(body(src(H['b25632_32k'])),'b25632w2_32k',W(2)))
for n,w in [('r12832',None),('r12832',4),('r12864',None),('r12864',3),('r25632',None),('r25632',2),('r12864_8k',3),('r25632_8k',2)]:
    nm='rr'+n[1:]+('w%d'%w if w else '')
    add(nm,rename(rr(body(src(H[n]))),nm,W(w) if w else ""))
gu=body(src('042b288f665b')); dn=body(src('465e63cd84ab')); dn6=body(src('ee1805eda914'))
add('gu',rename(gu,'gu')); add('gu_ng',rename(strip_guard(gu),'gu_ng')); add('gu_ne',rename(no_epi(gu),'gu_ne'))
add('dn',rename(dn,'dn')); add('dn_ng',rename(strip_guard(dn),'dn_ng')); add('dn_ne',rename(no_epi(dn),'dn_ne')); add('dn6',rename(dn6,'dn6'))
add('dn_ngne',rename(no_epi(strip_guard(dn)),'dn_ngne'))
if len(sys.argv)>1:
    pre=body(src(sys.argv[1])); add('gu_pre',rename(pre,'gu_pre')); add('gu_prenng',rename(strip_guard(pre),'gu_prenng'))
hdr=src('49a84b7117c4'); hdr=hdr[:hdr.index('extern "C"')]
micro=r'''
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
extern "C" __global__ void __launch_bounds__(256) bar_only(float* __restrict__ C, int nk) {
  for (int k=0;k<nk;++k){ __syncthreads(); }
  C[(int)threadIdx.x]=1.f;
}
extern "C" __global__ void __launch_bounds__(256) lds_r(float* __restrict__ C, int nk) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  char* L=(char*)buf_dyn_shmem; int t=(int)threadIdx.x;
  uint4 v=make_uint4((uint)t,(uint)t,(uint)t,(uint)t);
  #pragma unroll
  for (int i=0;i<8;++i) *(uint4*)(L+i*4096+t*16)=v;
  uint4 x=v;
  for (int k=0;k<nk;++k){
    __syncthreads();
    #pragma unroll
    for (int i=0;i<16;++i){ uint4 y=*(uint4*)(L+(i&7)*4096+t*16); x.x^=y.x; x.y^=y.y; x.z^=y.z; x.w^=y.w; }
  }
  C[t]=(float)(x.x^x.y^x.z^x.w);
}
extern "C" __global__ void __launch_bounds__(256) lds_w(float* __restrict__ C, int nk) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  char* L=(char*)buf_dyn_shmem; int t=(int)threadIdx.x;
  for (int k=0;k<nk;++k){
    uint4 v=make_uint4((uint)k,(uint)t,(uint)k,(uint)t);
    #pragma unroll
    for (int i=0;i<8;++i) *(uint4*)(L+i*4096+t*16)=v;
    __syncthreads();
  }
  C[t]=*(float*)(L+t*16);
}
// per K-step: nv global 16-byte loads per thread for A and nv for B (the base 128 row tile pattern), no LDS, no math
extern "C" __global__ void __launch_bounds__(256) gl8(half_t* __restrict__ A, half_t* __restrict__ B, float* __restrict__ C, int K, int nk) {
  int t=(int)threadIdx.x, bx=(int)blockIdx.x, by=(int)blockIdx.y;
  uint4 x=make_uint4(0u,0u,0u,0u);
  for (int k=0;k<nk;++k){
    #pragma unroll
    for (int v=0;v<4;++v){ uint4 y=*(uint4*)(A+by*128*K+v*32*K+(t>>3)*K+k*64+(t&7)*8); x.x^=y.x; x.y^=y.y; x.z^=y.z; x.w^=y.w; }
    #pragma unroll
    for (int v=0;v<4;++v){ uint4 y=*(uint4*)(B+bx*128*K+v*32*K+(t>>3)*K+k*64+(t&7)*8); x.x^=y.x; x.y^=y.y; x.z^=y.z; x.w^=y.w; }
  }
  C[t]=(float)(x.x^x.y^x.z^x.w);
}
// same data volume, but 16 scratch dword traffic: staged via a private array, per K-step 8 x 16 bytes written and read back
extern "C" __global__ void __launch_bounds__(256) scr8(float* __restrict__ C, int nk, int stride) {
  volatile uint pr[32]; int t=(int)threadIdx.x; uint x=0;
  for (int k=0;k<nk;++k){
    for (int i=0;i<32;++i) pr[i]=(uint)k;
    for (int i=0;i<32;++i) x^=pr[i];
  }
  C[t]=(float)x;
}
'''
open('kernels.cu','w').write(hdr+"\n".join(s for _,s in K)+micro)
print(len(K),[n for n,_ in K])
