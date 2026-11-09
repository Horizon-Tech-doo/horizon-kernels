# Hand-scheduled K loop with v_mfma_f32_32x32x8f16, the instruction hipBLASLt uses, numerically correct.
# Any K order is correct if A and B use the same one: a lane reads 16 B at (row = l&31, K byte = (l>>5)*16) plus ki*32.
# Windows are 64 cycles against 32 for 16x16x16, so a memory op fits between two MFMAs. Usage as gen_h7.py. CH must be 64.
import sys
name,M,N,K,BM,BN,CH=sys.argv[1],*map(int,sys.argv[2:8]); out=sys.argv[8]; opts=dict(o.split('=') for o in sys.argv[9:])
assert CH==64
ROWB=CH*2; TA=BM*ROWB; IM=BM//64; JN=BN//64; N16=CH//16; TPR=2*N16; NVA=N16*BM//128; NVB=N16*BN//128; NCH=K//CH
VROW=(128//N16)*K; LROW=(128//N16)*ROWB
V0=8; XA=V0; XB=XA+IM*4; YA=XB+JN*4; YB=YA+IM*4; ST=YB+JN*4; VEND=ST+(NVA+NVB)*4
def vt(b,n=4): return f"v[{b}:{b+n-1}]"
def bank(y): return (YA,YB) if y else (XA,XB)
L=[]
def e(s): L.append(s)
ncl=IM*JN
def mf(ki):
    a,b=bank(ki&1); r=[]
    for kp in range(2):
        for i in range(IM):
            for j in range(JN):
                r.append(f"v_mfma_f32_32x32x8f16 %[c{i*JN+j}], {vt(b+j*4+kp*2,2)}, {vt(a+i*4+kp*2,2)}, %[c{i*JN+j}]")
    return r
def rd(ki):
    a,b=bank(ki&1); r=[]
    for i in range(IM): r.append(f"ds_read_b128 {vt(a+i*4)}, %[lra] offset:{i*32*ROWB+ki*32}")
    for j in range(JN): r.append(f"ds_read_b128 {vt(b+j*4)}, %[lrb] offset:{TA+j*32*ROWB+ki*32}")
    return r
S=[]
for v in range(NVA): S.append((f"ds_write_b128 %[lw], {vt(ST+v*4)} offset:{v*LROW}",f"buffer_load_dwordx4 {vt(ST+v*4)}, %[voa], s[60:63], {'0' if v==0 else f's{68+v}'} offen"))
for v in range(NVB): S.append((f"ds_write_b128 %[lw], {vt(ST+(NVA+v)*4)} offset:{TA+v*LROW}",f"buffer_load_dwordx4 {vt(ST+(NVA+v)*4)}, %[vob], s[64:67], {'0' if v==0 else f's{68+v}'} offen"))
nS=len(S)
def bump():
    e("s_cmp_gt_u32 s76, 2"); e("s_cselect_b32 s77, %[stp], 0")
    e("s_add_u32 s60, s60, s77"); e("s_addc_u32 s61, s61, 0"); e("s_add_u32 s64, s64, s77"); e("s_addc_u32 s65, s65, 0")
e("s_nop 4")
for i,x in enumerate(['%[a0]','%[a1]','%[a2]','%[a3]','%[b0]','%[b1]','%[b2]','%[b3]']): e(f"s_mov_b32 s{60+i}, {x}")
for v in range(1,8): e(f"s_mov_b32 s{68+v}, %[so{v}]")
e("s_mov_b32 s76, %[cnt]")
for w,l in S: e(l)
e("s_waitcnt vmcnt(0)")
for w,l in S: e(w)
bump()
for w,l in S: e(l)
e("s_waitcnt lgkmcnt(0)"); e("s_barrier")
for r in rd(0): e(r)
e("LOOP%=:")
W=len(mf(0))                 # MFMAs per ki
BA=int(float(opts.get('ba',0.0))*W)   # in ki 2: barrier A after this many MFMAs; reads of ki3 go before it
FB=int(float(opts.get('fb',0.5))*W)   # in ki 3: barrier B after this many MFMAs; writes run from barrier A to it
gapr=max(1,W//len(rd(0))); 
for ki in range(4):
    e("s_waitcnt lgkmcnt(0)")
    ms=mf(ki)
    if ki<2:
        R=rd(ki+1); ri=0
        for idx,m in enumerate(ms):
            e(m)
            if ri<len(R) and idx%gapr==0: e(R[ri]); ri+=1
        while ri<len(R): e(R[ri]); ri+=1
    elif ki==2:
        # reads of ki3 first, then barrier A, then the first half of the writes
        R=rd(3); ri=0; nw=nS//2; si=0; gw=max(1,(W-len(R))//max(1,nw))
        for idx,m in enumerate(ms):
            e(m)
            if ri<len(R): e(R[ri]); ri+=1
            elif ri==len(R):
                e("s_waitcnt lgkmcnt(0)"); e("s_barrier"); ri+=1; c=idx
                bump()
            elif si<nw and (idx-c)%gw==0:
                w,l=S[si]; e(f"s_waitcnt vmcnt({nS-1})"); e(w); e(l); si+=1
    else:
        rest=S[nS//2:]; R=rd(0); si=0; ri=0; gw=max(1,FB//max(1,len(rest))); bdone=False
        for idx,m in enumerate(ms):
            e(m)
            if idx<FB:
                if si<len(rest) and idx%gw==0:
                    w,l=rest[si]; e(f"s_waitcnt vmcnt({nS-1})"); e(w); e(l); si+=1
            else:
                if not bdone:
                    while si<len(rest):
                        w,l=rest[si]; e(f"s_waitcnt vmcnt({nS-1})"); e(w); e(l); si+=1
                    e("s_waitcnt lgkmcnt(0)"); e("s_barrier"); bdone=True
                if ri<len(R): e(R[ri]); ri+=1
        while ri<len(R): e(R[ri]); ri+=1
e("s_sub_u32 s76, s76, 1"); e("s_cmp_eq_u32 s76, 0"); e("s_cbranch_scc0 LOOP%=")
e("s_waitcnt vmcnt(0) lgkmcnt(0)"); e("s_nop 7"); e("s_nop 7"); e("s_nop 7"); e("s_nop 7")
asm="\\n\\t".join(L)
outs=", ".join(f'[c{i}] "+a"(c{i})' for i in range(ncl))
ins=[f'[a{i}] "s"(ra[{i}])' for i in range(4)]+[f'[b{i}] "s"(rb[{i}])' for i in range(4)]+[f'[so{v}] "s"((uint)({v*VROW*2}))' for v in range(1,8)]+[f'[stp] "s"((uint)({CH*2}))',f'[cnt] "s"((uint)({NCH}))','[voa] "v"(voa)','[vob] "v"(vob)','[lw] "v"(lw)','[lra] "v"(lra)','[lrb] "v"(lrb)']
cl=[f'"v{i}"' for i in range(V0,VEND)]+[f'"s{i}"' for i in list(range(60,68))+list(range(69,78))]+['"scc"','"memory"']
SW=opts.get('sw','')
SWLINE='const int by=(int)blockIdx.y,bx=(int)blockIdx.x;' if not SW else 'const dim3 d3_=tl::rasterization2D%s<%s>(); const int by=(int)d3_.y,bx=(int)d3_.x;'%('Row' if SW[0]=='r' else 'Column',SW[1:])
src=f'''#include "k1_common.h"
#include <tl_templates/hip/threadblock_swizzle.h>
typedef float f32x16_ __attribute__((ext_vector_type(16)));
extern "C" __global__ void __launch_bounds__(256) {name}(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {{
  const int t=(int)threadIdx.x,l=t&63; const int wm={'(t&127)>>6' if BN==128 else 't>>7'}, wn={'t>>7' if BN==128 else '(t&127)>>6'}; {SWLINE}
  const i32x4_t ra=make_wave_buffer_resource(A), rb=make_wave_buffer_resource(B);
  const uint voa=(uint)(2*((by*{BM}+t/{TPR})*{K}+(t%{TPR})*8)), vob=(uint)(2*((bx*{BN}+t/{TPR})*{K}+(t%{TPR})*8));
  const uint lw=(uint)((t/{TPR})*{ROWB}+(t%{TPR})*16), lra=(uint)((wm*{BM//2}+(t&31))*{ROWB}+(l>>5)*16), lrb=(uint)((wn*{BN//2}+(t&31))*{ROWB}+(l>>5)*16);
  {" ".join(f"f32x16_ c{i}={{0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f}};" for i in range(ncl))}
  asm volatile("{asm}" : {outs} : {", ".join(ins)} : {", ".join(cl)});
  f32x16_ cc[{ncl}]={{{",".join(f"c{i}" for i in range(ncl))}}};
  #pragma unroll
  for (int i5=0;i5<{ncl};++i5) {{
    f32x16_ v_=cc[i5];
    #pragma unroll
    for (int q=0;q<4;++q) {{
      uint2 u;
      *((half_t*)(&(((half2*)(&(u.x)))->x)))=(half_t)(v_[q*4+0]); *((half_t*)(&(((half2*)(&(u.x)))->y)))=(half_t)(v_[q*4+1]);
      *((half_t*)(&(((half2*)(&(u.y)))->x)))=(half_t)(v_[q*4+2]); *((half_t*)(&(((half2*)(&(u.y)))->y)))=(half_t)(v_[q*4+3]);
      *(uint2*)(C+(((by*{BM}+wm*{BM//2}+(i5/{JN})*32+(t&31))*{N})+(bx*{BN}+wn*{BN//2}+(i5%{JN})*32+((t&63)>>5)*16+q*4)))=u;
    }}
  }}
}}
'''
open(out,'w').write(src)
