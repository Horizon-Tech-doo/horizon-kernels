# Hand-scheduled K loop in one inline asm block: the Tensile PGR2 layout with v_mfma_f32_16x16x16f16.
# Memory ops are spread between MFMAs, barriers do not stop math, accumulators in AGPRs.
# Usage: gen_h7.py <name> M N K BM BN CH [out.cu] [opts]. Row-major LDS. CH must be 64.
import sys
name,M,N,K,BM,BN,CH=sys.argv[1],*map(int,sys.argv[2:8]); out=sys.argv[8]; opts=dict(o.split('=') for o in sys.argv[9:])
FR=float(opts.get('fr',0.25))   # fraction of the last ki windows used for reading the next tile's ki0
RDIV=int(opts.get('rdiv',2))    # 12 reads of ki1 spread over W/RDIV windows of ki0
assert CH==64
ROWB=CH*2+int(opts.get('pad',0)); TA=BM*ROWB; TB=BN*ROWB; IM=BM//32; JN=BN//32; N16=CH//16; TPR=2*N16; NVA=N16*BM//128; NVB=N16*BN//128; NCH=K//CH; NKI=CH//32
VROW=(128//N16)*K; LROW=(128//N16)*ROWB
V0=8
XA=V0; XB=XA+IM*4; YA=XB+JN*4; YB=YA+IM*4; ST=YB+JN*4; VEND=ST+(NVA+NVB)*4
def vt(b,n=4): return f"v[{b}:{b+n-1}]"
def bank(y): return (YA,YB) if y else (XA,XB)
L=[]
def e(s): L.append(s)
def mf(ki):
    a,b=bank(ki&1); r=[]
    for kp in range(2):
        for i in range(IM):
            for j in range(JN):
                r.append(f"v_mfma_f32_16x16x16f16 %[c{i*JN+j}], {vt(b+j*4+kp*2,2)}, {vt(a+i*4+kp*2,2)}, %[c{i*JN+j}]")
    return r
def rd(ki):
    y=ki&1; a,b=bank(y); r=[]
    for i in range(IM): r.append(f"ds_read_b128 {vt(a+i*4)}, %[lra] offset:{i*16*ROWB+ki*64}")
    for j in range(JN): r.append(f"ds_read_b128 {vt(b+j*4)}, %[lrb] offset:{TA+j*16*ROWB+ki*64}")
    return r
def staged():
    r=[]
    for v in range(NVA): r.append((ST+v*4,f"ds_write_b128 %[lw], {vt(ST+v*4)} offset:{v*LROW}",f"buffer_load_dwordx4 {vt(ST+v*4)}, %[voa], s[60:63], {'0' if v==0 else f's{68+v}'} offen"))
    for v in range(NVB): r.append((ST+(NVA+v)*4,f"ds_write_b128 %[lw], {vt(ST+(NVA+v)*4)} offset:{TA+v*LROW}",f"buffer_load_dwordx4 {vt(ST+(NVA+v)*4)}, %[vob], s[64:67], {'0' if v==0 else f's{68+v}'} offen"))
    return r
STG=int(opts.get('stg',0)); STGS=int(opts.get('stgs',2)); STGM=opts.get('stgm','a')
def bump():
    if STG:   # K start stagger: the load pointer wraps at the end of K, so the unused last load stays in range
        e("s_add_u32 s60, s60, %[stp]"); e("s_addc_u32 s61, s61, 0"); e("s_add_u32 s64, s64, %[stp]"); e("s_addc_u32 s65, s65, 0")
        e("s_sub_u32 s78, s78, 1"); e("s_cmp_eq_u32 s78, 0"); e("s_cselect_b32 s77, %[wrp], 0")
        e("s_sub_u32 s60, s60, s77"); e("s_subb_u32 s61, s61, 0"); e("s_sub_u32 s64, s64, s77"); e("s_subb_u32 s65, s65, 0")
        e("s_cmp_eq_u32 s78, 0"); e("s_cselect_b32 s78, %[cnt], s78")
    else:
        e("s_cmp_gt_u32 s76, 2"); e("s_cselect_b32 s77, %[stp], 0")
        e("s_add_u32 s60, s60, s77"); e("s_addc_u32 s61, s61, 0"); e("s_add_u32 s64, s64, s77"); e("s_addc_u32 s65, s65, 0")
# entry
e("s_nop 4")
for i,x in enumerate(['%[a0]','%[a1]','%[a2]','%[a3]','%[b0]','%[b1]','%[b2]','%[b3]']): e(f"s_mov_b32 s{60+i}, {x}")
for v in range(1,8): e(f"s_mov_b32 s{68+v}, %[so{v}]")
e("s_mov_b32 s76, %[cnt]")
if STG: e("s_mov_b32 s78, %[s78i]")
S=staged(); nS=len(S)
for _,w,l in S: e(l)
e("s_waitcnt vmcnt(0)")
for _,w,l in S: e(w)
bump()
for _,w,l in S: e(l)
e("s_waitcnt lgkmcnt(0)"); e("s_barrier")
for r in rd(0): e(r)
e("LOOP%=:")
e("s_waitcnt lgkmcnt(0)")
m0=mf(0); W=len(m0); R1=rd(1); gap=max(1,(W//RDIV)//len(R1)); ri=0
for idx,m in enumerate(m0):
    e(m)
    if ri<len(R1) and idx%gap==0: e(R1[ri]); ri+=1
while ri<len(R1): e(R1[ri]); ri+=1
e("s_waitcnt lgkmcnt(0)"); e("s_barrier")
m1=mf(1); W1=len(m1); wN=int(W1*(1-FR)); gapw=max(1,wN//nS); si=0; R0=rd(0); nrd=W1-wN; ri=0; bumped=False; barr=False
gapr=max(1,nrd//len(R0))
for idx,m in enumerate(m1):
    e(m)
    if si<nS and idx%gapw==0:
        if si==0: bump()
        _,w,l=S[si]; e(f"s_waitcnt vmcnt({nS-1})"); e(w); e(l); si+=1
    if idx>=wN:
        if not barr:
            while si<nS:
                _,w,l=S[si]; e(f"s_waitcnt vmcnt({nS-1})"); e(w); e(l); si+=1
            e("s_waitcnt lgkmcnt(0)"); e("s_barrier"); barr=True
        if (idx-wN)%gapr==0 and ri<len(R0): e(R0[ri]); ri+=1
while ri<len(R0): e(R0[ri]); ri+=1
e("s_sub_u32 s76, s76, 1"); e("s_cmp_eq_u32 s76, 0"); e("s_cbranch_scc0 LOOP%=")
e("s_waitcnt vmcnt(0) lgkmcnt(0)"); e("s_nop 7"); e("s_nop 7"); e("s_nop 7")
asm="\\n\\t".join(L)
ncl=(BM*BN)//1024
outs=", ".join(f'[c{i}] "+a"(c{i})' for i in range(ncl))
ins=[f'[a{i}] "s"(ra[{i}])' for i in range(4)]+[f'[b{i}] "s"(rb[{i}])' for i in range(4)]+[f'[so{v}] "s"((uint)({v*VROW*2}))' for v in range(1,8)]+[f'[stp] "s"((uint)({CH*2}))',f'[wrp] "s"((uint)({NCH*CH*2}))','[s78i] "s"((uint)('+str(NCH)+'-s0_))',f'[cnt] "s"((uint)({NCH}))','[voa] "v"(voa)','[vob] "v"(vob)','[lw] "v"(lw)','[lra] "v"(lra)','[lrb] "v"(lrb)']
cl=[f'"v{i}"' for i in range(V0,VEND)]+[f'"s{i}"' for i in list(range(60,68))+list(range(69,79))]+['"scc"','"memory"']
SW=opts.get('sw','')
SWLINE='const int by=(int)blockIdx.y,bx=(int)blockIdx.x;' if not SW else 'const dim3 d3_=tl::rasterization2D%s<%s>(); const int by=(int)d3_.y,bx=(int)d3_.x;'%('Row' if SW[0]=='r' else 'Column',SW[1:])
STG0='0' if not STG else '((%s)&%d)*%d'%({'a':'bx+by','x':'bx','y':'by'}[STGM],STG-1,STGS)
src=f'''#include "k1_common.h"
#include <tl_templates/hip/threadblock_swizzle.h>
extern "C" __global__ void __launch_bounds__(256) {name}(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {{
  const int t=(int)threadIdx.x,l=t&63; const int wm={'(t&127)>>6' if BN==128 else 't>>7'}, wn={'t>>7' if BN==128 else '(t&127)>>6'}; {SWLINE}
  const int s0_={STG0}; const i32x4_t ra=make_wave_buffer_resource(A+s0_*{CH}), rb=make_wave_buffer_resource(B+s0_*{CH});
  const uint voa=(uint)(2*((by*{BM}+t/{TPR})*{K}+(t%{TPR})*8)), vob=(uint)(2*((bx*{BN}+t/{TPR})*{K}+(t%{TPR})*8));
  const uint lw=(uint)((t/{TPR})*{ROWB}+(t%{TPR})*16), lra=(uint)((wm*{BM//2}+(t&15))*{ROWB}+((l>>4)&3)*16), lrb=(uint)((wn*{BN//2}+(t&15))*{ROWB}+((l>>4)&3)*16);
  {" ".join(f"float32x4 c{i}={{0.f,0.f,0.f,0.f}};" for i in range(ncl))}
  asm volatile("{asm}" : {outs} : {", ".join(ins)} : {", ".join(cl)});
  float32x4 cc[{ncl}]={{{",".join(f"c{i}" for i in range(ncl))}}};
  #pragma unroll
  for (int i5=0;i5<{ncl};++i5) {{
    float32x4 v_=cc[i5]; half_t h[4]; uint2 u;
    *((half_t*)(&(((half2*)(&(u.x)))->x)))=(half_t)(v_[0]); *((half_t*)(&(((half2*)(&(u.x)))->y)))=(half_t)(v_[1]);
    *((half_t*)(&(((half2*)(&(u.y)))->x)))=(half_t)(v_[2]); *((half_t*)(&(((half2*)(&(u.y)))->y)))=(half_t)(v_[3]);
    *(uint2*)(C+(((by*{BM}+wm*{BM//2}+(i5/{JN})*16+(t&15))*{N})+(bx*{BN}+wn*{BN//2}+(i5%{JN})*16+((t&63)>>4)*4)))=u;
  }}
}}
'''
open(out,'w').write(src)
