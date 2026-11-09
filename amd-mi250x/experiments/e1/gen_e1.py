# build solo TUs (one kernel each) and a combined TU. Text transforms are the same as analysis/mb/gen.py.
import re,glob,os,sys,json
R="/scratch/project_XXXXXXXXX/tilelang/cache/0.1.15/linux-x86_64/kernels/"
def body(s): return s[s.index('extern "C"'):]
def hdr(s): return s[:s.index('extern "C"')]
def rename(s,name,attr=""):
    return re.sub(r'__launch_bounds__\(256\)\s+\w+\(', '__launch_bounds__(256) '+(attr+' ' if attr else '')+name+'(', s)
def rr(s):
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
W=lambda n:"__attribute__((amdgpu_waves_per_eu(%d)))"%n
SH={'4k':(4096,4096,4096),'8k':(8192,8192,8192),'32k':(32768,1024,2048),'17k':(4096,17408,5120)}
# variant name -> base src key, tile, transforms, attr
V={'b12864':('b12864',128,64,0,None),'b25632w2':('b25632',256,32,0,2),'rr12864w3':('r12864',128,64,1,3),'rr25632w2':('r25632',256,32,1,2),
   # no-attribute controls: is the attribute needed when compiled alone?
   'b25632':('b25632',256,32,0,None),'rr25632':('r25632',256,32,1,None),'rr12864':('r12864',128,64,1,None)}
os.makedirs('solo',exist_ok=True)
h=hdr(open('src/b12864_4k.cu').read())
tbl=[];comb=[]
for vn,(src,bM,bK,isrr,w) in V.items():
  for sn,(M,N,K) in SH.items():
    s=body(open(f'src/{src}_{sn}.cu').read())
    if isrr: s=rr(s)
    name=f'{vn}_{sn}'
    s=rename(s,name,W(w) if w else "")
    open(f'solo/{name}.cu','w').write(h+s)
    comb.append(s)
    dyn=32768 if bM==128 else 24576
    tbl.append(dict(name=name,variant=vn,shape=sn,M=M,N=N,K=K,bM=bM,bN=128,bK=bK,dyn=dyn))
open('combined.cu','w').write(h+"\n".join(comb))
json.dump(tbl,open('table.json','w'))
print(len(tbl))
# check against cached kernels used by MB
H={'b12864_4k':'df817aeb23d4','b25632_4k':'28b00176afce','r12864_4k':'d0a6d9c6d104','r25632_4k':'49685536bd2c','b12864_8k':'07c5beaa2519','b25632_8k':'c5ceacd5e92d','r12864_8k':'460994032ab3','r25632_8k':'aea8968bb747','b12864_32k':'52676ec39801','b25632_32k':'2fab2a272ffa'}
for k,hh in H.items():
    a=open(glob.glob(R+hh+"*")[0]+"/device_kernel.cu").read(); b=open(f'src/{k}.cu').read()
    print(k,'same as MB cached source:',a==b)
# ---- step 6 + clock instrumented + 32x32x8 timing-only
PAIRS=[3200,3072,5632,5888,4224,2816,2944]
V6={'b12864':('b12864',128,64,0,None),'b25632w2':('b25632',256,32,0,2),'rr12864w3':('r12864',128,64,1,3),'rr25632w2':('r25632',256,32,1,2)}
for vn,(src,bM,bK,isrr,w) in V6.items():
  for N in PAIRS:
    s=body(open(f'src/{src}_n{N}.cu').read())
    if isrr: s=rr(s)
    name=f'{vn}_n{N}'
    s=rename(s,name,W(w) if w else "")
    open(f'solo/{name}.cu','w').write(h+s)
TRDEF='struct Tr { unsigned long long re,r0,r1,me,m0,m1; unsigned hw; unsigned bk; };\n'
def ck(s,gx):
    # signature: add trace arg; head stamps; tail stamps before epilogue; write at the very end
    s=re.sub(r'half_t\* __restrict__ C\) \{','half_t* __restrict__ C, Tr* __restrict__ tr) {\n  unsigned long long re_=__builtin_amdgcn_s_memrealtime(), me_=__builtin_amdgcn_s_memtime(); unsigned hw_=(unsigned)__builtin_amdgcn_s_getreg(63492);',s,1)
    idx=s.rfind('#pragma unroll',0,s.index('half_t C_local_cast'))
    s=s[:idx]+'unsigned long long r1_=__builtin_amdgcn_s_memrealtime(), m1_=__builtin_amdgcn_s_memtime();\n  '+s[idx:]
    end=s.rstrip().rfind('}')
    tail='if ((((int)threadIdx.x)&63)==0){ unsigned long long r2_=__builtin_amdgcn_s_memrealtime(), m2_=__builtin_amdgcn_s_memtime(); int bi_=(int)blockIdx.y*%d+(int)blockIdx.x; Tr* p=tr+bi_*4+(((int)threadIdx.x)>>6); p->re=re_;p->r0=r1_;p->r1=r2_;p->me=me_;p->m0=m1_;p->m1=m2_;p->hw=hw_;p->bk=bi_; }\n'%gx
    return s[:end]+tail+s[end:]
tblx=[]
os.makedirs('ck',exist_ok=True)
for vn,(src,bM,bK,isrr,w) in V6.items():
  for sn,(M,N,K) in SH.items():
    s=body(open(f'src/{src}_{sn}.cu').read())
    if isrr: s=rr(s)
    s=rename(s,f'{vn}_{sn}_ck',W(w) if w else "")
    s=ck(s,N//128)
    open(f'ck/{vn}_{sn}_ck.cu','w').write(h+TRDEF+s)
  for N in [3200,3072]:
    s=body(open(f'src/{src}_n{N}.cu').read())
    if isrr: s=rr(s)
    s=rename(s,f'{vn}_n{N}_ck',W(w) if w else "")
    s=ck(s,N//128)
    open(f'ck/{vn}_n{N}_ck.cu','w').write(h+TRDEF+s)
def m32(s):
    p=re.compile(r'for \(int (kp(?:_\d+)?) = 0; \1 < 2; \+\+\1\) \{\s*for \(int (i(?:_\d+)?) = 0; \2 < 4; \+\+\2\) \{\s*for \(int (j(?:_\d+)?) = 0; \3 < 4; \+\+\3\) \{\s*\{.*?\};\s*\}\s*\}\s*\}',re.S)
    assert p.search(s)
    return p.sub('for (int kp_x = 0; kp_x < 4; ++kp_x) { for (int i_x = 0; i_x < 2; ++i_x) { for (int j_x = 0; j_x < 2; ++j_x) { *(((float32x16_*)C_l) + ((i_x * 2) + j_x)) = __builtin_amdgcn_mfma_f32_32x32x8f16(*(((float16x4*)B_local) + ((j_x * 4) + kp_x)), *(((float16x4*)A_local) + ((i_x * 4) + kp_x)), *(((float32x16_*)C_l) + ((i_x * 2) + j_x)), 0, 0, 0); } } }',s)
TD='typedef float float32x16_ __attribute__((ext_vector_type(16)));\n'
for vn,src,w,isrr in [('b12864m32','b12864',None,0),('rr12864m32w3','r12864',3,1),('rr12864m32','r12864',None,1)]:
  for sn in ['4k','8k','32k','17k']:
    s=body(open(f'src/{src}_{sn}.cu').read())
    if isrr: s=rr(s)
    s=m32(s)
    s=rename(s,f'{vn}_{sn}',W(w) if w else "")
    open(f'solo/{vn}_{sn}.cu','w').write(h+TD+s)
print('extra done')
