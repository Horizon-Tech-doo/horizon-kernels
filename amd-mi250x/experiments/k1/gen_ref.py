# Reference kernels from the shared TileLang cache, patched exactly as analysis/mb/gen.py did, one TU per kernel.
import re,glob,sys,os
R="/scratch/project_XXXXXXXXX/tilelang/cache/0.1.15/linux-x86_64/kernels/"
def src(h): return open(glob.glob(R+h+"*")[0]+"/device_kernel.cu").read()
def body(s): return s[s.index('extern "C"'):]
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
H={'b12832':'49a84b7117c4','b12864':'df817aeb23d4','r12832':'e4d16c90dc8c','r12864':'d0a6d9c6d104','b12864_8k':'07c5beaa2519','r12864_8k':'460994032ab3'}
hdr=src('49a84b7117c4'); hdr=hdr[:hdr.index('extern "C"')]
K={'b12832':rename(body(src(H['b12832'])),'b12832'),
   'b12864':rename(body(src(H['b12864'])),'b12864'),
   'b12864_8k':rename(body(src(H['b12864_8k'])),'b12864_8k'),
   'rr12832':rename(rr(body(src(H['r12832']))),'rr12832'),
   'rr12864w3':rename(rr(body(src(H['r12864']))),'rr12864w3',W(3)),
   'rr12864_8kw3':rename(rr(body(src(H['r12864_8k']))),'rr12864_8kw3',W(3))}
os.makedirs('cu',exist_ok=True)
for n,s in K.items(): open('cu/ref_%s.cu'%n,'w').write(hdr+s)
print(list(K))
